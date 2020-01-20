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

package org.apache.lucene.search;

import java.io.IOException;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.ivfflat.ImmutableUnClusterableVector;
import org.apache.lucene.util.ivfflat.IvfFlatCacheReader;
import org.apache.lucene.util.ivfflat.SortedImmutableVectorValue;

public class KnnIvfFlatScoreWeight extends ConstantScoreWeight {
  private final String field;

  private final ScoreMode scoreMode;

  private final float[] queryVector;

  private final int ef;

  private final int numCentroids;

  public KnnIvfFlatScoreWeight(Query query, float score, String field, ScoreMode scoreMode, float[] queryVector, int ef, int numCentroids) {
    super(query, score);
    this.field = field;
    this.scoreMode = scoreMode;
    this.queryVector = queryVector;
    this.ef = ef;
    this.numCentroids = numCentroids;
  }

  /**
   * Returns a {@link Scorer} which can iterate in order over all matching
   * documents and assign them a score.
   * <p>
   * <b>NOTE:</b> null can be returned if no documents will be scored by this
   * query.
   * <p>
   * <b>NOTE</b>: The returned {@link Scorer} does not have
   * {@link LeafReader#getLiveDocs()} applied, they need to be checked on top.
   *
   * @param context the {@link LeafReaderContext} for which to return the {@link Scorer}.
   * @return a {@link Scorer} which scores documents in/out-of order.
   * @throws IOException if there is a low-level I/O error
   */
  @Override
  public Scorer scorer(LeafReaderContext context) throws IOException {
    ScorerSupplier supplier = scorerSupplier(context);
    if (supplier == null) {
      return null;
    }
    return supplier.get(Long.MAX_VALUE);
  }

  @Override
  public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
    final FieldInfo fi = context.reader().getFieldInfos().fieldInfo(field);
    int numDimensions = fi.getVectorNumDimensions();
    if (numDimensions != queryVector.length) {
      throw new IllegalArgumentException("field=\"" + field + "\" was indexed with dimensions=" + numDimensions + "; this is incompatible with query dimensions=" + queryVector.length);
    }

    final IvfFlatCacheReader ivfFlatCacheReader = new IvfFlatCacheReader(field, context);
    final VectorValues vectorValues = context.reader().getVectorValues(field);
    if (vectorValues == null) {
      // No docs in this segment/field indexed any vector values
      return null;
    }

    final Weight weight = this;
    return new ScorerSupplier() {
      @Override
      public Scorer get(long leadCost) throws IOException {
        SortedImmutableVectorValue neighbors = ivfFlatCacheReader.search(queryVector, ef, numCentroids, vectorValues);
        return new Scorer(weight) {

          int doc = -1;
          float score = 0.0f;
          int size = neighbors.size();
          int offset = 0;

          @Override
          public DocIdSetIterator iterator() {
            return new DocIdSetIterator() {
              @Override
              public int docID() {
                return doc;
              }

              @Override
              public int nextDoc() throws IOException {
                return advance(offset);
              }

              @Override
              public int advance(int target) throws IOException {
                if (target > size || neighbors.size() == 0) {
                  doc = NO_MORE_DOCS;
                  score = 0.0f;
                } else {
                  while (offset < target) {
                    neighbors.pop();
                    offset++;
                  }
                  ImmutableUnClusterableVector next = neighbors.pop();
                  offset++;
                  if (next == null) {
                    doc = NO_MORE_DOCS;
                    score = 0.0f;
                  } else {
                    doc = next.docId();
                    score = 1.0F / (next.distance() / numDimensions + 0.01F);
                  }
                }
                return doc;
              }

              @Override
              public long cost() {
                return size;
              }
            };
          }

          @Override
          public float getMaxScore(int upTo) throws IOException {
            return Float.POSITIVE_INFINITY;
          }

          @Override
          public float score() throws IOException {
            return score;
          }

          @Override
          public int docID() {
            return doc;
          }
        };
      }

      @Override
      public long cost() {
        return ef;
      }
    };
  }

  /**
   * @param ctx
   * @return {@code true} if the object can be cached against a given leaf
   */
  @Override
  public boolean isCacheable(LeafReaderContext ctx) {
    return true;
  }
}
