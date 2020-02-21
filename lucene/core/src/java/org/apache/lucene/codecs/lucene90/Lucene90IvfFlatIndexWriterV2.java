/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.lucene90;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.IvfFlatWriter;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.IntsRef;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Writes vector values and ivfflat index to index segments.
 */
public class Lucene90IvfFlatIndexWriterV2 extends IvfFlatIndexWriter {
  private final IndexOutput ivfFlatMeta, ivfFlatData, vectorData;

  private boolean finished;

  public Lucene90IvfFlatIndexWriterV2(SegmentWriteState state) throws IOException {
    assert state.fieldInfos.hasIvfFlatAndVectorValues();

    final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormatV2.META_EXTENSION);

    final String ivfDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormatV2.IVF_DATA_EXTENSION);

    final String vecDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90KnnGraphFormat.VECTOR_DATA_EXTENSION);

    boolean success = false;
    try {
      this.ivfFlatMeta = state.directory.createOutput(metaFileName, state.context);

      this.ivfFlatData = state.directory.createOutput(ivfDataFileName, state.context);

      this.vectorData = state.directory.createOutput(vecDataFileName, state.context);

      CodecUtil.writeIndexHeader(ivfFlatMeta, Lucene90IvfFlatIndexFormatV2.META_CODEC_NAME,
          Lucene90IvfFlatIndexFormatV2.VERSION_CURRENT, state.segmentInfo.getId(),
          state.segmentSuffix);

      CodecUtil.writeIndexHeader(ivfFlatData, Lucene90IvfFlatIndexFormatV2.IVF_DATA_CODEC_NAME,
          Lucene90IvfFlatIndexFormatV2.VERSION_CURRENT, state.segmentInfo.getId(),
          state.segmentSuffix);

      CodecUtil.writeIndexHeader(vectorData, Lucene90KnnGraphFormat.VECTOR_DATA_CODEC_NAME,
          Lucene90IvfFlatIndexFormat.VERSION_CURRENT, state.segmentInfo.getId(),
          state.segmentSuffix);

      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Write all values contained in the provided reader.
   */
  @Override
  public void writeField(FieldInfo fieldInfo, IvfFlatIndexReader reader) throws IOException {
    long vectorDataOffset = vectorData.getFilePointer();

    final Map<Integer, Integer> vecToDocOffset = writeVectors(fieldInfo, reader);

    long ivfDataOffset = ivfFlatData.getFilePointer();

    final Map<Integer, Long> ivfOffsets = writeData(fieldInfo, reader, vecToDocOffset);

    writeMeta(fieldInfo, vectorDataOffset, vectorData.getFilePointer() - vectorDataOffset,
        ivfDataOffset, ivfFlatData.getFilePointer() - ivfDataOffset, ivfOffsets);
  }

  private Map<Integer, Integer> writeVectors(final FieldInfo fieldInfo, final IvfFlatIndexReader reader) throws IOException {
    int numDims = fieldInfo.getVectorNumDimensions();

    final VectorValues vectors = reader.getVectorValues(fieldInfo.name);

    final Map<Integer, Integer> vecToDocOffset = new HashMap<>();
    int offset = 0;
    for (int doc = vectors.nextDoc(); doc != NO_MORE_DOCS; doc = vectors.nextDoc()) {
      writeVectorValue(numDims, vectors);
      vecToDocOffset.put(doc, offset++);
    }

    return vecToDocOffset;
  }

  private void writeVectorValue(int numDims, final VectorValues vectors) throws IOException {
    // write vector value
    BytesRef binaryValue = vectors.binaryValue();
    VectorValues.verifyNumDimensions(binaryValue.length, numDims);
    vectorData.writeBytes(binaryValue.bytes, binaryValue.offset, binaryValue.length);
  }

  private void writeMeta(final FieldInfo field, long vectorDataOffset, long vectorDataLength, long ivfDataOffset,
                         long ivfDataLenght, final Map<Integer, Long> ivfOffsets) throws IOException {
    ivfFlatMeta.writeInt(field.number);
    ivfFlatMeta.writeVLong(vectorDataOffset);
    ivfFlatMeta.writeVLong(vectorDataLength);
    ivfFlatMeta.writeVLong(ivfDataOffset);
    ivfFlatMeta.writeVLong(ivfDataLenght);

    int numClusters = ivfOffsets.size();
    ivfFlatMeta.writeInt(numClusters);

    for (Map.Entry<Integer, Long> cluster: ivfOffsets.entrySet()) {
      ivfFlatMeta.writeVInt(cluster.getKey());
      ivfFlatMeta.writeVLong(cluster.getValue());
    }
  }

  private Map<Integer, Long> writeData(final FieldInfo fieldInfo, final IvfFlatIndexReader reader,
                                       final Map<Integer, Integer> vecToDocOffset) throws IOException {
    final IvfFlatValues ivfFlatValues = reader.getIvfFlatValues(fieldInfo.name);
    int[] centroids = ivfFlatValues.getCentroids();

    final Map<Integer, Long> ivfOffsets = new HashMap<>();
    for (int centroid : centroids) {
      ivfOffsets.put(centroid, ivfFlatData.getFilePointer());

      writeIvfData(ivfFlatValues.getIvfLink(centroid), vecToDocOffset);
    }

    return ivfOffsets;
  }

  private void writeIvfData(final IntsRef ivfFlatLink, final Map<Integer, Integer> vecToDocOffset) throws IOException {
    ivfFlatData.writeInt(ivfFlatLink.length);
    if (ivfFlatLink.length > 0) {
      int stop = ivfFlatLink.offset + ivfFlatLink.length;
      // sort friend ids
      Arrays.sort(ivfFlatLink.ints, ivfFlatLink.offset, stop);
      // write the smallest friend id
      int ivfElemId = ivfFlatLink.ints[ivfFlatLink.offset];
      ivfFlatData.writeVInt(ivfElemId);
      /// write its order in segment
      ivfFlatData.writeVInt(vecToDocOffset.get(ivfElemId));
      for (int idx = ivfFlatLink.offset + 1; idx < stop; ++idx) {
        // write delta
        assert ivfFlatLink.ints[idx] > ivfFlatLink.ints[idx - 1];
        ivfFlatMeta.writeVInt(ivfFlatLink.ints[idx] - ivfFlatLink.ints[idx - 1]);
        ivfFlatMeta.writeVInt(vecToDocOffset.get(ivfFlatLink.ints[idx]));
      }
    }
  }

  /**
   * Called once at the end before close
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }

    finished = true;

    if (ivfFlatMeta != null) {
      ivfFlatMeta.writeInt(-1);
      CodecUtil.writeFooter(ivfFlatMeta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
    }
    if (ivfFlatData != null) {
      CodecUtil.writeFooter(ivfFlatData);
    }
  }

  @Override
  protected void mergeOneField(FieldInfo fieldInfo, MergeState state) throws IOException {
    int readerLength = state.ivfFlatIndexReaders.length;
    final List<Lucene90KnnGraphWriter.VectorValuesSub> subs = new ArrayList<>(readerLength);
    for (int i = 0; i < readerLength; ++i) {
      final IvfFlatIndexReader ivfFlatIndexReader = state.ivfFlatIndexReaders[i];
      if (ivfFlatIndexReader != null) {
        subs.add(new Lucene90KnnGraphWriter.VectorValuesSub(i, state.docMaps[i],
            ivfFlatIndexReader.getVectorValues(fieldInfo.name)));
      }
    }

    final IvfFlatWriter ivfFlatWriter = new IvfFlatWriter(fieldInfo, Counter.newCounter());
    for (Lucene90KnnGraphWriter.VectorValuesSub sub : subs) {
      final MergeState.DocMap docMap = state.docMaps[sub.segmentIndex];
      int docId;
      while ((docId = sub.nextDoc()) != NO_MORE_DOCS) {
        int mappedDocId = docMap.get(docId);
        if (mappedDocId == -1) {
          continue;
        }

        assert sub.values.docID() == docId;
        ivfFlatWriter.addValue(mappedDocId, sub.values.binaryValue());
      }
    }

    ivfFlatWriter.flush(this);
  }

  /**
   * Closes this stream and releases any system resources associated
   * with it. If the stream is already closed then invoking this
   * method has no effect.
   *
   * <p> As noted in {@link AutoCloseable#close()}, cases where the
   * close may fail require careful attention. It is strongly advised
   * to relinquish the underlying resources and to internally
   * <em>mark</em> the {@code Closeable} as closed, prior to throwing
   * the {@code IOException}.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void close() throws IOException {
    IOUtils.close(ivfFlatMeta, ivfFlatData, vectorData);
  }
}
