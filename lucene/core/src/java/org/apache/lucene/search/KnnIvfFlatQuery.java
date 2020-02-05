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
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Approximate nearest neighbor search query for high dimensional vector values
 * using ivfflat.
 */
public class KnnIvfFlatQuery extends Query implements Accountable {
  private static final int DEFAULT_RESERVED_CENTER_POINTS_FOR_SEARCH = 32;

  private static final int DEFAULT_RESULTS_RESERVED = 50;

  private final String field;

  private final float[] queryVector;

  private final int k;

  private final int nprobe;

  private final long ramBytesUsed;

  /**
   * Constructor
   *
   * @param field the field name of query
   * @param queryVector target vector
   */
  public KnnIvfFlatQuery(String field, float[] queryVector) {
    this(field, queryVector, DEFAULT_RESULTS_RESERVED, DEFAULT_RESERVED_CENTER_POINTS_FOR_SEARCH);
  }

  /**
   * Constructor.
   *
   * @param field the field name of query
   * @param queryVector target vector
   * @param k the number of top docs to reserve
   */
  public KnnIvfFlatQuery(String field, float[] queryVector, int k) {
    this(field, queryVector, k, DEFAULT_RESERVED_CENTER_POINTS_FOR_SEARCH);
  }

  /**
   * Constructor.
   *
   * @param field the field name of query
   * @param queryVector target vector
   * @param k the number of top docs to reserve
   * @param nprobe the number of clusters for similarity searching
   */
  public KnnIvfFlatQuery(String field, float[] queryVector, int k, int nprobe) {
    this.field = field;
    this.queryVector = queryVector;
    this.k = k;
    this.nprobe = nprobe;
    this.ramBytesUsed = RamUsageEstimator.shallowSizeOfInstance(getClass());
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
    return new KnnIvfFlatScoreWeight(this, boost, scoreMode, field, queryVector, k, nprobe);
  }

  /**
   * Prints a query to a string, with <code>field</code> assumed to be the
   * default field and omitted.
   *
   * @param field field for serialize to String
   */
  @Override
  public String toString(String field) {
    return String.format(Locale.ROOT, "KnnIvfFlatQuery{field=%s;fromQuery=%s;numCentroids=%d}",
        field, Arrays.toString(queryVector), nprobe);
  }

  /**
   * Recurse through the query tree, visiting any child queries
   *
   * @param visitor a QueryVisitor to be called by each query in the tree
   */
  @Override
  public void visit(QueryVisitor visitor) {
    if (visitor.acceptField(field)) {
      visitor.visitLeaf(this);
    }
  }

  /**
   * Override and implement query instance equivalence properly in a subclass.
   * This is required so that {@link QueryCache} works properly.
   * <p>
   * Typically a query will be equal to another only if it's an instance of
   * the same class and its document-filtering properties are identical that other
   * instance. Utility methods are provided for certain repetitive code.
   *
   * @param other target object for comparison
   * @see #sameClassAs(Object)
   * @see #classHash()
   */
  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) &&
        equalsTo(getClass().cast(other));
  }

  /**
   * Override and implement query hash code properly in a subclass.
   * This is required so that {@link QueryCache} works properly.
   *
   * @see #equals(Object)
   */
  @Override
  public int hashCode() {
    return classHash() + Objects.hash(field, nprobe, queryVector);
  }

  /**
   * Return the memory usage of this object in bytes. Negative values are illegal.
   */
  @Override
  public long ramBytesUsed() {
    return this.ramBytesUsed;
  }

  private boolean equalsTo(KnnIvfFlatQuery other) {
    return Objects.equals(field, other.field) &&
        Arrays.equals(queryVector, other.queryVector) &&
        Objects.equals(nprobe, other.nprobe) &&
        Objects.equals(k, other.k);
  }
}
