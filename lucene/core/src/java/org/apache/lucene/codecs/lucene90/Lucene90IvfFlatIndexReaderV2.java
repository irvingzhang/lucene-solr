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
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.RamUsageEstimator;

public class Lucene90IvfFlatIndexReaderV2 extends Lucene90IvfFlatIndexReader {
  private final IndexInput ivfFlatData;

  public Lucene90IvfFlatIndexReaderV2(SegmentReadState state) throws IOException {
    super(state);
    this.ramBytesUsed = RamUsageEstimator.shallowSizeOfInstance(Lucene90IvfFlatIndexReaderV2.class);

    final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormat.META_EXTENSION);

    int metaVersion = readMeta(state, metaFileName);

    this.ivfFlatData = openAndCheckFile(state, metaVersion, Lucene90IvfFlatIndexFormat.IVF_DATA_EXTENSION,
        Lucene90IvfFlatIndexFormat.IVF_DATA_CODEC_NAME);
  }

  /**
   * Checks consistency of this reader.
   * <p>
   * Note that this may be costly in terms of I/O, e.g.
   * may involve computing a checksum value against large data files.
   *
   * @lucene.internal
   */
  @Override
  public void checkIntegrity() throws IOException {
    super.checkIntegrity();

    if (ivfFlatData != null) {
      CodecUtil.checksumEntireFile(ivfFlatData);
    }
  }

  /**
   * Returns the {@link IvfFlatValues} for the given {@code field}
   *
   * @param field the field name for retrieval
   */
  @Override
  public IvfFlatValues getIvfFlatValues(String field) {
    final IvfFlatEntryV2 ivfFlatEntry = (IvfFlatEntryV2) ivfFlats.get(field);
    if (ivfFlatEntry == null) {
      return IvfFlatValues.EMPTY;
    }

    return new IndexedIvfFlatReaderV2(maxDoc, ivfFlatEntry, ivfFlatData);
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
    IOUtils.close(this.ivfFlatData);
  }

  protected void readFields(final ChecksumIndexInput meta, final FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      final FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }

      long vectorDataOffset = meta.readVLong();
      long vectorDataLength = meta.readVLong();
      long ivfDataOffset = meta.readVLong();
      long ivfDataLength = meta.readVLong();
      int clustersSize = meta.readInt();

      final Map<Integer, Long> clusters = new HashMap<>(clustersSize);
      final Map<Integer, Integer> docToOrd = new HashMap<>();
      for (int i = 0; i < clustersSize; ++i) {
        int centroid = meta.readVInt();
        docToOrd.put(centroid, meta.readVInt());
        clusters.put(centroid, meta.readVLong());
      }

      int docSize = meta.readInt();
      for (int i = 0; i < docSize; ++i) {
        docToOrd.put(meta.readVInt(), meta.readVInt());
      }

      IvfFlatEntryV2 ivfFlatEntry = new IvfFlatEntryV2(vectorDataOffset, vectorDataLength,
          ivfDataOffset, ivfDataLength, docToOrd, clusters);

      ramBytesUsed += RamUsageEstimator.shallowSizeOfInstance(IvfFlatEntryV2.class);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToVectorOrd);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToCentroidOffset);
      ivfFlats.put(info.name, ivfFlatEntry);
    }
  }

  private static final class IvfFlatEntryV2 extends VectorDataEntry {
    final int[] centroids;
    final long ivfDataOffset;
    final long ivfDataLenght;
    final Map<Integer, Long> docToCentroidOffset;

    public IvfFlatEntryV2(long vectorDataOffset, long vectorDataLength, long ivfDataOffset, long ivfDataLenght,
                          final Map<Integer, Integer> docToVectorOrd, final Map<Integer, Long> docToCentroidOffset) {
      super(vectorDataOffset, vectorDataLength, docToVectorOrd);
      this.ivfDataOffset = ivfDataOffset;
      this.ivfDataLenght = ivfDataLenght;
      this.centroids = docToCentroidOffset.keySet().stream().mapToInt(i -> i).toArray();
      this.docToCentroidOffset = docToCentroidOffset;
    }
  }

  /** Read the ivfflat index hole in memory */
  private static final class IndexedIvfFlatReaderV2 extends IvfFlatValues {
    final long maxDoc;

    final IndexInput ivfFlatData;

    final IvfFlatEntryV2 ivfFlatEntry;

    private IndexedIvfFlatReaderV2(long maxDoc, final IvfFlatEntryV2 ivfFlatEntry, final IndexInput ivfFlatData) {
      this.maxDoc = maxDoc;
      this.ivfFlatData = ivfFlatData;
      this.ivfFlatEntry = ivfFlatEntry;
    }

    /**
     * Returns all the center points.
     *
     * @return the center points of all clusters
     */
    @Override
    public int[] getCentroids() {
      return ivfFlatEntry.centroids;
    }

    /**
     * Returns the inverse list (doc ID list) that belongs to the {@code centroid}.
     *
     * @param centroid the specified centroid
     * @return points of the specified centroid if the specified centroid exists, empty {@link IntsRef} otherwise.
     */
    @Override
    public IntsRef getIvfLink(int centroid) throws IOException {
      if (centroid < 0 || maxDoc < centroid) {
        throw new IllegalArgumentException("centroid must be >= 0 or <= maxDocID (=" + maxDoc + ")");
      }

      if (!ivfFlatEntry.docToCentroidOffset.containsKey(centroid)) {
        return new IntsRef();
      }

      long offset = ivfFlatEntry.docToCentroidOffset.get(centroid);
      ivfFlatData.seek(offset);
      int numClusterPoints = ivfFlatData.readInt();

      int[] clusterPoints = new int[numClusterPoints];
      for (int num = 0; num < numClusterPoints; ++num) {
        clusterPoints[num] = ivfFlatData.readVInt();
      }

      return new IntsRef(clusterPoints, 0, clusterPoints.length);
    }
  }
}
