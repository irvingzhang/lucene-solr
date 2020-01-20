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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.IntsRef;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Writes vector values and ivfflat index to index segments.
 */
public class Lucene90IvfFlatIndexWriter extends IvfFlatIndexWriter {
  private final IndexOutput ivfFlatMeta, vectorData;

  private boolean finished;

  public Lucene90IvfFlatIndexWriter(SegmentWriteState state) throws IOException {
    assert state.fieldInfos.hasIvfFlatAndVectorValues();

    final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormat.META_EXTENSION);

    final String vecDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90KnnGraphFormat.VECTOR_DATA_EXTENSION);

    boolean success = false;
    try {
      this.ivfFlatMeta = state.directory.createOutput(metaFileName, state.context);

      this.vectorData = state.directory.createOutput(vecDataFileName, state.context);

      CodecUtil.writeIndexHeader(ivfFlatMeta, Lucene90IvfFlatIndexFormat.META_CODEC_NAME,
          Lucene90IvfFlatIndexFormat.VERSION_CURRENT, state.segmentInfo.getId(),
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

    writeMeta(fieldInfo, vectorDataOffset, vectorData.getFilePointer() - vectorDataOffset,
        fieldInfo, reader, vecToDocOffset);
  }

  private Map<Integer, Integer> writeVectors(FieldInfo fieldInfo, IvfFlatIndexReader reader) throws IOException {
    int numDims = fieldInfo.getVectorNumDimensions();

    VectorValues vectors = reader.getVectorValues(fieldInfo.name);

    final Map<Integer, Integer> vecToDocOffset = new HashMap<>();
    int offset = 0;
    for (int doc = vectors.nextDoc(); doc != NO_MORE_DOCS; doc = vectors.nextDoc()) {
      writeVectorValue(numDims, vectors);
      vecToDocOffset.put(doc, offset++);
    }

    return vecToDocOffset;
  }

  private void writeVectorValue(int numDims, VectorValues vectors) throws IOException {
    // write vector value
    BytesRef binaryValue = vectors.binaryValue();
    VectorValues.verifyNumDimensions(binaryValue.length, numDims);
    vectorData.writeBytes(binaryValue.bytes, binaryValue.offset, binaryValue.length);
  }

  private void writeMeta(FieldInfo field, long vectorDataOffset, long vectorDataLength, FieldInfo fieldInfo, IvfFlatIndexReader reader,
                         Map<Integer, Integer> vecToDocOffset) throws IOException {
    ivfFlatMeta.writeInt(field.number);
    ivfFlatMeta.writeVLong(vectorDataOffset);
    ivfFlatMeta.writeVLong(vectorDataLength);

    final IvfFlatValues ivfFlatValues = reader.getIvfFlatValues(fieldInfo.name);

    int[] centroids = ivfFlatValues.getCentroids();
    ivfFlatMeta.writeInt(centroids.length);

    for (int centroid : centroids) {
      ivfFlatMeta.writeVInt(centroid);
      IntsRef ivfFlatLink = ivfFlatValues.getIvfLink(centroid);
      ivfFlatMeta.writeInt(ivfFlatLink.length);
      if (ivfFlatLink.length > 0) {
        int stop = ivfFlatLink.offset + ivfFlatLink.length;
        // sort friend ids
        Arrays.sort(ivfFlatLink.ints, ivfFlatLink.offset, stop);
        // write the smallest friend id
        int ivfElemId = ivfFlatLink.ints[ivfFlatLink.offset];
        ivfFlatMeta.writeVInt(ivfElemId);
        ivfFlatMeta.writeVInt(vecToDocOffset.get(ivfElemId));
        for (int idx = ivfFlatLink.offset + 1; idx < stop; ++idx) {
          // write delta
          assert ivfFlatLink.ints[idx] > ivfFlatLink.ints[idx - 1];
          ivfFlatMeta.writeVInt(ivfFlatLink.ints[idx] - ivfFlatLink.ints[idx - 1]);
          ivfFlatMeta.writeVInt(vecToDocOffset.get(ivfFlatLink.ints[idx]));
        }
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
  }

  @Override
  protected void mergeOneField(FieldInfo fieldInfo, MergeState state) {
    /// TODO
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
    IOUtils.close(ivfFlatMeta, vectorData);
  }
}
