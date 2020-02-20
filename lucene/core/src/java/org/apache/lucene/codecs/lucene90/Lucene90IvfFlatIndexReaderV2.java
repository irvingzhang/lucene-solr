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
import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.RamUsageEstimator;

public class Lucene90IvfFlatIndexReaderV2 extends IvfFlatIndexReader {
  private final FieldInfos fieldInfos;

  private final IndexInput ivfFlatData, vectorData;

  private final int maxDoc;

  private long ramBytesUsed;

  private final Map<String, IvfFlatEntryV2> ivfFlats = new HashMap<>();

  public Lucene90IvfFlatIndexReaderV2(SegmentReadState state) throws IOException {
    this.fieldInfos = state.fieldInfos;
    this.maxDoc = state.segmentInfo.maxDoc();
    this.ramBytesUsed = RamUsageEstimator.shallowSizeOfInstance(Lucene90IvfFlatIndexReaderV2.class);

    final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormatV2.META_EXTENSION);

    int metaVersion = readMeta(state, metaFileName);

    this.vectorData = openAndCheckFile(state, metaVersion, Lucene90KnnGraphFormat.VECTOR_DATA_EXTENSION,
        Lucene90KnnGraphFormat.VECTOR_DATA_CODEC_NAME);

    this.ivfFlatData = openAndCheckFile(state, metaVersion, Lucene90IvfFlatIndexFormatV2.IVF_DATA_EXTENSION,
        Lucene90IvfFlatIndexFormatV2.IVF_DATA_CODEC_NAME);
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
    if (vectorData != null) {
      CodecUtil.checksumEntireFile(vectorData);
    }

    if (ivfFlatData != null) {
      CodecUtil.checksumEntireFile(ivfFlatData);
    }
  }

  /**
   * Returns the {@link VectorValues} for the given {@code field}
   *
   * @param field the field name for retrieval
   */
  @Override
  public VectorValues getVectorValues(String field) throws IOException {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    if (info == null) {
      return VectorValues.EMPTY;
    }

    int numDims = info.getVectorNumDimensions();
    final Lucene90KnnGraphReader.VectorDataEntry entry = ivfFlats.get(field);
    if (numDims <= 0 || entry == null) {
      return VectorValues.EMPTY;
    }

    final IndexInput bytesSlice = vectorData.slice("vector-data",
        entry.vectorDataOffset, entry.vectorDataLength);

    return new Lucene90KnnGraphReader.RandomAccessVectorValuesReader(maxDoc, numDims, entry, bytesSlice);
  }

  /**
   * Returns the {@link IvfFlatValues} for the given {@code field}
   *
   * @param field the field name for retrieval
   */
  @Override
  public IvfFlatValues getIvfFlatValues(String field) throws IOException {
    final IvfFlatEntryV2 ivfFlatEntry = ivfFlats.get(field);
    if (ivfFlatEntry == null) {
      return IvfFlatValues.EMPTY;
    }

    final IndexInput bytesSlice = ivfFlatData.slice("knn-ivf-data", ivfFlatEntry.ivfDataOffset,
        ivfFlatEntry.ivfDataLenght);

    return new IndexedIvfFlatReaderV2(maxDoc, ivfFlatEntry, bytesSlice);
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
    IOUtils.close(this.vectorData, this.ivfFlatData);
  }

  /**
   * Return the memory usage of this object in bytes. Negative values are illegal.
   */
  @Override
  public long ramBytesUsed() {
    return this.ramBytesUsed;
  }

  private IndexInput openAndCheckFile(final SegmentReadState state, int metaVersion,
                                      final String extension, final String codecName) throws IOException {
    final String dataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);

    boolean success = false;
    IndexInput dataIn = null;
    try {
      dataIn = state.directory.openInput(dataFileName, state.context);

      int vectorDataVersion = CodecUtil.checkIndexHeader(dataIn, codecName,
          Lucene90IvfFlatIndexFormat.VERSION_START, Lucene90IvfFlatIndexFormat.VERSION_CURRENT,
          state.segmentInfo.getId(), state.segmentSuffix);

      if (metaVersion != vectorDataVersion) {
        throw new CorruptIndexException("Format versions mismatch: meta=" + metaVersion + ", vector data=" + vectorDataVersion, vectorData);
      }

      CodecUtil.retrieveChecksum(dataIn);

      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(dataIn);
        dataIn = null;
      }
    }

    return dataIn;
  }

  private int readMeta(final SegmentReadState state, final String metaFileName) throws IOException {
    int metaVersion = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName, state.context)) {
      Throwable priorE = null;
      try {
        metaVersion = CodecUtil.checkIndexHeader(meta, Lucene90IvfFlatIndexFormat.META_CODEC_NAME,
            Lucene90IvfFlatIndexFormat.VERSION_START, Lucene90IvfFlatIndexFormat.VERSION_CURRENT,
            state.segmentInfo.getId(), state.segmentSuffix);

        readFields(meta, state.fieldInfos);
      } catch (IOException exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
    }

    return metaVersion;
  }

  private void readFields(final ChecksumIndexInput meta, final FieldInfos infos) throws IOException {
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
        int numClusterPoints = ivfFlatData.readInt();
        int[] clusterPoints = new int[numClusterPoints];
        clusterPoints[0] = meta.readVInt();
        docToOrd.put(clusterPoints[0], ivfFlatData.readVInt());
        for (int num = 1; num < numClusterPoints; ++num) {
          clusterPoints[num] = ivfFlatData.readVInt() + clusterPoints[num - 1];
          docToOrd.put(clusterPoints[num], ivfFlatData.readVInt());
        }

        clusters.put(centroid, meta.readVLong());
      }

      IvfFlatEntryV2 ivfFlatEntry = new IvfFlatEntryV2(vectorDataOffset, vectorDataLength,
          ivfDataOffset, ivfDataLength, docToOrd, clusters);

      ramBytesUsed += RamUsageEstimator.shallowSizeOfInstance(IvfFlatEntryV2.class);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToVectorOrd);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToCentroidOffset);
      ivfFlats.put(info.name, ivfFlatEntry);
    }
  }

  private static final class IvfFlatEntryV2 extends Lucene90KnnGraphReader.VectorDataEntry {
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

      ivfFlatData.seek(ivfFlatEntry.docToCentroidOffset.get(centroid));
      int numClusterPoints = ivfFlatData.readInt();
      int[] clusterPoints = new int[numClusterPoints];
      clusterPoints[0] = ivfFlatData.readVInt();
      for (int num = 1; num < numClusterPoints; ++num) {
        clusterPoints[num] = ivfFlatData.readVInt() + clusterPoints[num - 1];
      }

      return new IntsRef(clusterPoints, 0, clusterPoints.length);
    }
  }
}
