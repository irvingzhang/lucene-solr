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

  private final int k;

  private final int nprobe;

  public KnnIvfFlatScoreWeight(Query query, float score, ScoreMode scoreMode, String field, float[] queryVector, int k, int nprobe) {
    super(query, score);
    this.field = field;
    this.scoreMode = scoreMode;
    this.queryVector = queryVector;
    this.k = k;
    this.nprobe = nprobe;
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
    final ScorerSupplier supplier = scorerSupplier(context);
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
      throw new IllegalArgumentException("field=\"" + field + "\" was indexed with dimensions=" +
          numDimensions + "; this is incompatible with query dimensions=" + queryVector.length);
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
        final SortedImmutableVectorValue clusteredVectors = ivfFlatCacheReader.search(queryVector, k, nprobe, vectorValues);
        return new Scorer(weight) {

          int doc = -1;
          float score = 0.0F;
          int size = clusteredVectors.size();
          int offset = 0;

          @Override
          public DocIdSetIterator iterator() {
            return new DocIdSetIterator() {
              @Override
              public int docID() {
                return doc;
              }

              @Override
              public int nextDoc() {
                return advance(offset);
              }

              @Override
              public int advance(int target) {
                if (target > size || clusteredVectors.size() == 0) {
                  doc = NO_MORE_DOCS;
                  score = 0.0F;
                } else {
                  while (offset < target) {
                    clusteredVectors.pop();
                    offset++;
                  }

                  final ImmutableUnClusterableVector next = clusteredVectors.pop();
                  offset++;
                  if (next == null) {
                    doc = NO_MORE_DOCS;
                    score = 0.0F;
                  } else {
                    doc = next.docId();
                    if (scoreMode.needsScores()) {
                      score = 1.0F / (next.distance() + Float.MIN_NORMAL);
                    }
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
          public float getMaxScore(int upTo) {
            if (scoreMode.needsScores()) {
              return Float.POSITIVE_INFINITY;
            }

            return 0.0F;
          }

          @Override
          public float score() {
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
        return k;
      }
    };
  }

  /**
   *
   * @param ctx context of leaf reader
   * @return {@code true} if the object can be cached against a given leaf
   */
  @Override
  public boolean isCacheable(LeafReaderContext ctx) {
    return true;
  }
}
