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
import java.util.Iterator;
import java.util.List;

import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.ivfflat.Centroid;
import org.apache.lucene.util.ivfflat.ImmutableClusterableVector;
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

  public void addValue(int docID, BytesRef binaryValue) {
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

  public void flush(IvfFlatIndexWriter ivfFlatIndexWriter) throws IOException {
    ivfFlatCacheWriter.finish();
    float[][] rawVectors = ivfFlatCacheWriter.rawVectorsArray();

    final VectorValues vectorValues = new BufferedVectorValues(
        docsWithFieldVec.iterator(), rawVectors);

    ivfFlatIndexWriter.writeField(fieldInfo, new IvfFlatIndexReader() {
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
       */
      @Override
      public void close() {

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
      public void checkIntegrity() {

      }

      /**
       * Returns the {@link VectorValues} for the given {@code field}
       *
       * @param field fieldInfo name
       */
      @Override
      public VectorValues getVectorValues(String field) {
        return vectorValues;
      }

      /**
       * Returns the {@link IvfFlatValues} for the given {@code field}
       *
       * @param field fieldInfo name
       */
      @Override
      public IvfFlatValues getIvfFlatValues(String field) throws IOException {
        final List<ImmutableClusterableVector> immutableClusterableVectors = new ArrayList<>(rawVectors.length);
        final DocIdSetIterator docsWithField = docsWithFieldVec.iterator();
        int idx = 0;
        for (int doc = docsWithField.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = docsWithField.nextDoc()) {
          immutableClusterableVectors.add(new ImmutableClusterableVector(doc, rawVectors[idx++]));
        }

        final List<Centroid<ImmutableClusterableVector>> clusteredPoints = ivfFlatCacheWriter.cluster(immutableClusterableVectors);

        return new IvfFlatValues() {
          /**
           * Return the size of clusters.
           *
           * @return cluster size
           */
          @Override
          public int getClusterSize() {
            return clusteredPoints.size();
          }

          @Override
          public VectorValues getCentroids() {
            return new VectorValues() {
              int doc = -1;
              Centroid<ImmutableClusterableVector> current;
              Iterator<Centroid<ImmutableClusterableVector>> it = clusteredPoints.iterator();

              @Override
              public float[] vectorValue() throws IOException {
                return current.getCenter().getPoint();
              }

              @Override
              public boolean seek(int target) throws IOException {
                while (it.hasNext()) {
                  current = it.next();
                  if (current.getCenter().docId() == target) {
                    doc = target;
                    return true;
                  }
                }

                doc = NO_MORE_DOCS;
                return false;
              }

              @Override
              public int docID() {
                return doc;
              }

              @Override
              public int nextDoc() throws IOException {
                return advance(doc + 1);
              }

              @Override
              public int advance(int target) throws IOException {
                int _target = target;
                boolean found;
                do {
                  found = seek(_target++);
                } while (!found && doc != NO_MORE_DOCS);
                return doc;
              }

              @Override
              public long cost() {
                return clusteredPoints.size();
              }
            };
            /// return clusteredPoints.stream().mapToInt(i -> i.getCenter().docId()).toArray();
          }

          @Override
          public IntsRef getIvfLink(int centroid) {
            for (Centroid<ImmutableClusterableVector> clusteredPoint : clusteredPoints) {
              if (clusteredPoint.getCenter().docId() == centroid) {
                int[] ivfLink = clusteredPoint.getPoints().stream().mapToInt(ImmutableClusterableVector::docId).toArray();
                return new IntsRef(ivfLink, 0, ivfLink.length);
              }
            }

            return new IntsRef();
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

  static class BufferedVectorValues extends VectorValues {

    final DocIdSetIterator docsWithField;
    final float[][] vectorsArray;

    int bufferPos = 0;
    float[] value;

    BufferedVectorValues(DocIdSetIterator docsWithField, float[][] vectorsArray) {
      this.docsWithField = docsWithField;
      this.vectorsArray = vectorsArray;
    }

    @Override
    public float[] vectorValue() {
      return value;
    }

    @Override
    public int docID() {
      return docsWithField.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      int docID = docsWithField.nextDoc();
      if (docID != NO_MORE_DOCS) {
        value = vectorsArray[bufferPos++];
      }
      return docID;
    }

    @Override
    public int advance(int target) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean seek(int target) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return docsWithField.cost();
    }
  }
}
