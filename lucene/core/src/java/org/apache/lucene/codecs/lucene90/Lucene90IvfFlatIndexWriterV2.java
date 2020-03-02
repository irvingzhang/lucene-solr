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
public class Lucene90IvfFlatIndexWriterV2 extends Lucene90IvfFlatIndexWriter {
  private final IndexOutput ivfFlatData;

  public Lucene90IvfFlatIndexWriterV2(SegmentWriteState state) throws IOException {
    super(state);

    final String ivfDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormat.IVF_DATA_EXTENSION);

    boolean success = false;
    try {
      this.ivfFlatData = state.directory.createOutput(ivfDataFileName, state.context);

      CodecUtil.writeIndexHeader(ivfFlatData, Lucene90IvfFlatIndexFormat.IVF_DATA_CODEC_NAME,
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

    final Map<Integer, Long> ivfOffsets = writeData(fieldInfo, reader);

    writeMeta(fieldInfo, vectorDataOffset, vectorData.getFilePointer() - vectorDataOffset,
        ivfDataOffset, ivfFlatData.getFilePointer() - ivfDataOffset, ivfOffsets, vecToDocOffset);
  }

  private void writeMeta(final FieldInfo field, long vectorDataOffset, long vectorDataLength, long ivfDataOffset,
                         long ivfDataLenght, final Map<Integer, Long> ivfOffsets, final Map<Integer, Integer> vecToDocOffset) throws IOException {
    ivfFlatMeta.writeInt(field.number);
    ivfFlatMeta.writeVLong(vectorDataOffset);
    ivfFlatMeta.writeVLong(vectorDataLength);
    ivfFlatMeta.writeVLong(ivfDataOffset);
    ivfFlatMeta.writeVLong(ivfDataLenght);

    ivfFlatMeta.writeInt(ivfOffsets.size());

    for (Map.Entry<Integer, Long> cluster : ivfOffsets.entrySet()) {
      ivfFlatMeta.writeVInt(cluster.getKey());

      assert vecToDocOffset.containsKey(cluster.getKey());
      ivfFlatMeta.writeVInt(vecToDocOffset.get(cluster.getKey()));

      ivfFlatMeta.writeVLong(cluster.getValue());
    }

    ivfFlatMeta.writeInt(vecToDocOffset.size() - ivfOffsets.size());
    vecToDocOffset.entrySet().removeIf(i -> ivfOffsets.containsKey(i.getKey()));
    for (Map.Entry<Integer, Integer> offset : vecToDocOffset.entrySet()) {
      ivfFlatMeta.writeVInt(offset.getKey());
      ivfFlatMeta.writeVInt(offset.getValue());
    }
  }

  private Map<Integer, Long> writeData(final FieldInfo fieldInfo, final IvfFlatIndexReader reader) throws IOException {
    final IvfFlatValues ivfFlatValues = reader.getIvfFlatValues(fieldInfo.name);
    int[] centroids = ivfFlatValues.getCentroids();

    final Map<Integer, Long> ivfOffsets = new HashMap<>();
    for (int centroid : centroids) {
      ivfOffsets.put(centroid, ivfFlatData.getFilePointer());

      writeIvfData(ivfFlatValues.getIvfLink(centroid));
    }

    return ivfOffsets;
  }

  private void writeIvfData(final IntsRef ivfFlatLink) throws IOException {
    ivfFlatData.writeInt(ivfFlatLink.length);
    if (ivfFlatLink.length > 0) {
      int stop = ivfFlatLink.offset + ivfFlatLink.length;

      for (int idx = ivfFlatLink.offset; idx < stop; ++idx) {
        ivfFlatData.writeVInt(ivfFlatLink.ints[idx]);
      }
    }
  }

  /**
   * Called once at the end before close
   */
  @Override
  public void finish() throws IOException {
    super.finish();

    if (ivfFlatData != null) {
      CodecUtil.writeFooter(ivfFlatData);
    }
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
    super.close();
    IOUtils.close(ivfFlatData);
  }
}
