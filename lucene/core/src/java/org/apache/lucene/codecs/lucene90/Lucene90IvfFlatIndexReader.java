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

public class Lucene90IvfFlatIndexReader extends IvfFlatIndexReader {
  private final FieldInfos fieldInfos;

  private final IndexInput vectorData;

  private final int maxDoc;

  private long ramBytesUsed;

  private final Map<String, IvfFlatEntry> ivfFlats = new HashMap<>();

  public Lucene90IvfFlatIndexReader(SegmentReadState state) throws IOException {
    this.fieldInfos = state.fieldInfos;
    this.maxDoc = state.segmentInfo.maxDoc();
    this.ramBytesUsed = RamUsageEstimator.shallowSizeOfInstance(Lucene90IvfFlatIndexReader.class);

    final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name,
        state.segmentSuffix, Lucene90IvfFlatIndexFormat.META_EXTENSION);

    int versionMeta = readMetaAndFields(state, metaFileName);

    this.vectorData = openAndCheckFile(state, versionMeta, Lucene90KnnGraphFormat.VECTOR_DATA_EXTENSION,
        Lucene90KnnGraphFormat.VECTOR_DATA_CODEC_NAME);
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

    final IndexInput bytesSlice = vectorData.slice("vector-data", entry.vectorDataOffset, entry.vectorDataLength);
    return new Lucene90KnnGraphReader.RandomAccessVectorValuesReader(maxDoc, numDims, entry, bytesSlice);
  }

  /**
   * Returns the {@link IvfFlatValues} for the given {@code field}
   *
   * @param field the field name for retrieval
   */
  @Override
  public IvfFlatValues getIvfFlatValues(String field) {
    final IvfFlatEntry ivfFlatEntry = ivfFlats.get(field);
    if (ivfFlatEntry == null) {
      return IvfFlatValues.EMPTY;
    }

    return new IndexedIvfFlatReader(maxDoc, ivfFlatEntry);
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
    IOUtils.close(this.vectorData);
  }

  /**
   * Return the memory usage of this object in bytes. Negative values are illegal.
   */
  @Override
  public long ramBytesUsed() {
    return this.ramBytesUsed;
  }

  private IndexInput openAndCheckFile(SegmentReadState state, int versionMeta, String extension, String codecName) throws IOException {
    final String dataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);

    boolean success = false;
    IndexInput dataIn = null;
    try {
      dataIn = state.directory.openInput(dataFileName, state.context);

      int versionVectorData = CodecUtil.checkIndexHeader(dataIn, codecName,
          Lucene90IvfFlatIndexFormat.VERSION_START, Lucene90IvfFlatIndexFormat.VERSION_CURRENT,
          state.segmentInfo.getId(), state.segmentSuffix);

      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException("Format versions mismatch: meta=" + versionMeta + ", vector data=" + versionVectorData, vectorData);
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

  private int readMetaAndFields(SegmentReadState state, String metaFileName) throws IOException {
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName, state.context)) {
      Throwable priorE = null;
      try {
        versionMeta = CodecUtil.checkIndexHeader(meta, Lucene90IvfFlatIndexFormat.META_CODEC_NAME,
            Lucene90IvfFlatIndexFormat.VERSION_START, Lucene90IvfFlatIndexFormat.VERSION_CURRENT,
            state.segmentInfo.getId(), state.segmentSuffix);

        readFields(meta, state.fieldInfos);
      } catch (IOException exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
    }

    return versionMeta;
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }

      long vectorDataOffset = meta.readVLong();
      long vectorDataLength = meta.readVLong();
      int clustersSize = meta.readInt();

      final Map<Integer, IntsRef> clusters = new HashMap<>(clustersSize);
      final Map<Integer, Integer> docToOrd = new HashMap<>();
      for (int i = 0; i < clustersSize; ++i) {
        int cluster = meta.readVInt();
        int numClusterPoints = meta.readInt();
        int[] clusterPoints = new int[numClusterPoints];
        clusterPoints[0] = meta.readVInt();
        docToOrd.put(clusterPoints[0], meta.readVInt());
        for (int num = 1; num < numClusterPoints; ++num) {
          clusterPoints[num] = meta.readVInt() + clusterPoints[num - 1];
          docToOrd.put(clusterPoints[num], meta.readVInt());
        }

        clusters.put(cluster, new IntsRef(clusterPoints, 0, clusterPoints.length));
      }

      IvfFlatEntry ivfFlatEntry = new IvfFlatEntry(vectorDataOffset, vectorDataLength, docToOrd, clusters);

      ramBytesUsed += RamUsageEstimator.shallowSizeOfInstance(IvfFlatEntry.class);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToVectorOrd);
      ramBytesUsed += RamUsageEstimator.shallowSizeOf(ivfFlatEntry.docToCentroidOffset);
      ivfFlats.put(info.name, ivfFlatEntry);
    }
  }

  private static final class IvfFlatEntry extends Lucene90KnnGraphReader.VectorDataEntry {
    final int[] centroids;
    final Map<Integer, IntsRef> docToCentroidOffset;


    public IvfFlatEntry(long vectorDataOffset, long vectorDataLength, Map<Integer, Integer> docToVectorOrd,
                        Map<Integer, IntsRef> docToCentroidOffset) {
      super(vectorDataOffset, vectorDataLength, docToVectorOrd);
      this.centroids = docToCentroidOffset.keySet().stream().mapToInt(i -> i).toArray();
      this.docToCentroidOffset = docToCentroidOffset;
    }
  }

  /** Read the knn graph from the index input stream */
  private final static class IndexedIvfFlatReader extends IvfFlatValues {
    final long maxDoc;

    final IvfFlatEntry ivfFlatEntry;

    int doc = -1;

    private IndexedIvfFlatReader(long maxDoc, IvfFlatEntry ivfFlatEntry) {
      this.maxDoc = maxDoc;
      this.ivfFlatEntry = ivfFlatEntry;
    }

    /**
     * Returns the center points of the ivfflat index.
     *
     * @return the center points of all clusters
     */
    @Override
    public int[] getCentroids() {
      return ivfFlatEntry.centroids;
    }

    /**
     * Returns the inverse list (doc ID list) that belongs to the {@code centroid}.
     * Each doc ID in the list is closer to the current {@code centroid} than any other center points.
     * {@code centroid} must be a valid doc ID, ie. &ge; 0 and &le; {@link #NO_MORE_DOCS}.
     * It is illegal to call this method after {@link #advanceExact(int)}
     * returned {@code false}.
     *
     * @param centroid@return friend list
     */
    @Override
    public IntsRef getIvfLink(int centroid) {
      if (centroid < 0 || maxDoc < centroid) {
        throw new IllegalArgumentException("centroid must be >= 0 or <= maxDocID (=" + maxDoc + ")");
      }


      return ivfFlatEntry.docToCentroidOffset.getOrDefault(centroid, new IntsRef());
    }

    /**
     * Move the pointer to exactly {@code target} and return whether
     * {@code target} has friends lists.
     * {@code target} must be a valid doc ID, ie. &ge; 0 and &lt; {@code maxDoc}.
     * After this method returns, {@link #docID()} retuns {@code target}.
     *
     * @param target the target doc ID that advanced to
     */
    @Override
    public boolean advanceExact(int target) {
      advance(target);
      return doc == target;
    }

    /**
     * Returns the following:
     * <ul>
     * <li><code>-1</code> if {@link #nextDoc()} or
     * {@link #advance(int)} were not called yet.
     * <li>{@link #NO_MORE_DOCS} if the iterator has exhausted.
     * <li>Otherwise it should return the doc ID it is currently on.
     * </ul>
     * <p>
     *
     * @since 2.9
     */
    @Override
    public int docID() {
      return doc;
    }

    /**
     * Advances to the next document in the set and returns the doc it is
     * currently on, or {@link #NO_MORE_DOCS} if there are no more docs in the
     * set.<br>
     *
     * <b>NOTE:</b> after the iterator has exhausted you should not call this
     * method, as it may result in unpredicted behavior.
     *
     * @since 2.9
     */
    @Override
    public int nextDoc() {
      return advance(doc + 1);
    }

    /**
     * Advances to the first beyond the current whose document number is greater
     * than or equal to <i>target</i>, and returns the document number itself.
     * Exhausts the iterator and returns {@link #NO_MORE_DOCS} if <i>target</i>
     * is greater than the highest document number in the set.
     * <p>
     * The behavior of this method is <b>undefined</b> when called with
     * <code> target &le; current</code>, or after the iterator has exhausted.
     * Both cases may result in unpredicted behavior.
     * <p>
     * When <code> target &gt; current</code> it behaves as if written:
     *
     * <pre class="prettyprint">
     * int advance(int target) {
     *   int doc;
     *   while ((doc = nextDoc()) &lt; target) {
     *   }
     *   return doc;
     * }
     * </pre>
     * <p>
     * Some implementations are considerably more efficient than that.
     * <p>
     * <b>NOTE:</b> this method may be called with {@link #NO_MORE_DOCS} for
     * efficiency by some Scorers. If your implementation cannot efficiently
     * determine that it should exhaust, it is recommended that you check for that
     * value in each call to this method.
     * <p>
     *
     * @param target the target that advanced to
     * @since 2.9
     */
    @Override
    public int advance(int target) {
      if (target >= maxDoc) {
        return doc = NO_MORE_DOCS;
      }

      doc = target - 1;
      do {
        if (ivfFlatEntry.docToCentroidOffset.containsKey(++doc)) {
          break;
        }
      } while (doc < maxDoc);

      if (doc == maxDoc) {
        return doc = NO_MORE_DOCS;
      }

      return doc;
    }

    /**
     * Returns the estimated cost of this {@link org.apache.lucene.search.DocIdSetIterator}.
     * <p>
     * This is generally an upper bound of the number of documents this iterator
     * might match, but may be a rough heuristic, hardcoded value, or otherwise
     * completely inaccurate.
     */
    @Override
    public long cost() {
      return maxDoc;
    }
  }
}
