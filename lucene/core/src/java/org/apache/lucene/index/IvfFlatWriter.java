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

package org.apache.lucene.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.codecs.KnnGraphReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.ivfflat.Centroid;
import org.apache.lucene.util.ivfflat.Clusterable;
import org.apache.lucene.util.ivfflat.ImmutableClusterableVector;
import org.apache.lucene.util.ivfflat.IvFFlatIndex;
import org.apache.lucene.util.ivfflat.IvfFlatCacheWriter;

public class IvfFlatWriter implements Accountable {
  private final FieldInfo fieldInfo;
  private final Counter iwBytesUsed;
  private final DocsWithFieldSet docsWithFieldVec;
  private final IvfFlatCacheWriter ivfFlatCacheWriter;

  private int lastDocID = -1;

  private long bytesUsed = 0L;

  public IvfFlatWriter(FieldInfo fieldInfo, Counter iwBytesUsed) {
    this.fieldInfo = fieldInfo;
    this.iwBytesUsed = iwBytesUsed;
    this.docsWithFieldVec = new DocsWithFieldSet();
    this.ivfFlatCacheWriter = new IvfFlatCacheWriter(fieldInfo.getVectorNumDimensions(), fieldInfo.getVectorDistFunc());

    updateBytesUsed();
  }

  public void addValue(int docID, BytesRef binaryValue) throws IOException {
    if (docID <= lastDocID) {
      throw new IllegalArgumentException("VectorValuesField \"" + fieldInfo.name + "\" appears more than once in this document (only one value is allowed per field)");
    }
    if (binaryValue == null) {
      throw new IllegalArgumentException("field=\"" + fieldInfo.name + "\": null value not allowed");
    }

    ivfFlatCacheWriter.insert(docID, binaryValue);
    docsWithFieldVec.add(docID);

    updateBytesUsed();

    lastDocID = docID;
  }

  private void updateBytesUsed() {
    final long newBytesUsed = docsWithFieldVec.ramBytesUsed();
    if (iwBytesUsed != null) {
      iwBytesUsed.addAndGet(newBytesUsed - bytesUsed);
    }
    bytesUsed = newBytesUsed;
  }

  public void flush(Sorter.DocMap sortMap, IvfFlatIndexWriter ivfFlatIndexWriter) throws IOException {
    ivfFlatCacheWriter.finish();
    float[][] rawVectors = ivfFlatCacheWriter.rawVectorsArray();
    final VectorValues vectors = new KnnGraphValuesWriter.BufferedVectorValues(docsWithFieldVec.iterator(),
        rawVectors);

    final VectorValues vectorValues = vectors;

    ivfFlatIndexWriter.writeField(fieldInfo,
        new IvfFlatIndexReader() {

          /**
           * Return the memory usage of this object in bytes. Negative values are illegal.
           */
          @Override
          public long ramBytesUsed() {
            return 0L;
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

          }

          /**
           * Returns the {@link VectorValues} for the given {@code field}
           *
           * @param field
           */
          @Override
          public VectorValues getVectorValues(String field) throws IOException {
            return vectorValues;
          }

          /**
           * Returns the {@link IvfFlatValues} for the given {@code field}
           *
           * @param field
           */
          @Override
          public IvfFlatValues getIvfFlatValues(String field) throws IOException {
            final List<ImmutableClusterableVector> immutableClusterableVectors = new ArrayList<>();
            final DocIdSetIterator docsWithField = docsWithFieldVec.iterator();
            int idx = 0;
            for (int doc = docsWithField.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = docsWithField.nextDoc()) {
              immutableClusterableVectors.add(new ImmutableClusterableVector(doc, rawVectors[idx++]));
            }

            final List<IvFFlatIndex.ClusteredPoints> clusteredPoints = ivfFlatCacheWriter.cluster(immutableClusterableVectors);

            return new IvfFlatValues() {
              @Override
              public int[] getCentroids() {
                return clusteredPoints.stream().mapToInt(IvFFlatIndex.ClusteredPoints::getCenter).toArray();
              }

              @Override
              public IntsRef getIvfLink(int centroid) {
                for (IvFFlatIndex.ClusteredPoints clusteredPoint : clusteredPoints) {
                  if (clusteredPoint.getCenter() != centroid) {
                    continue;
                  } else {
                    int[] ivfLink = clusteredPoint.getPoints().stream().mapToInt(i -> i).toArray();
                    return new IntsRef(ivfLink, 0, ivfLink.length);
                  }
                }

                return new IntsRef();
              }

              @Override
              public boolean advanceExact(int target) throws IOException {
                int result = advance(target);
                return result == target;
              }

              @Override
              public int docID() {
                return NO_MORE_DOCS;
              }

              @Override
              public int nextDoc() throws IOException {
                return NO_MORE_DOCS;
              }

              @Override
              public int advance(int target) throws IOException {
                return target;
              }

              @Override
              public long cost() {
                return ivfFlatCacheWriter.rawVectorsArray().length;
              }
            };
          }
        });
  }

  /**
   * Return the memory usage of this object in bytes. Negative values are illegal.
   */
  @Override
  public long ramBytesUsed() {
    return bytesUsed;
  }
}
