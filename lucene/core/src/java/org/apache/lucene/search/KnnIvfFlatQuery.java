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
import java.util.Locale;
import java.util.Objects;

import org.apache.lucene.util.Accountable;

public class KnnIvfFlatQuery extends Query implements Accountable {
  private static final int DEFAULT_RESERVED_CENTER_POINTS_FOR_SEARCH = 10;

  private final String field;

  private final float[] queryVector;

  private final int ef;

  private final int numCentroids;

  private final long ramBytesUsed;

  public KnnIvfFlatQuery(String field, float[] queryVector, int ef, long ramBytesUsed) {
    this(field, queryVector, ef, DEFAULT_RESERVED_CENTER_POINTS_FOR_SEARCH, ramBytesUsed);
  }

  public KnnIvfFlatQuery(String field, float[] queryVector, int ef, int numCentroids, long bytesUsed) {
    this.field = field;
    this.queryVector = queryVector;
    this.ef = ef;
    this.numCentroids = numCentroids;
    this.ramBytesUsed = bytesUsed;
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
    return new KnnScoreWeight(this, boost, scoreMode, field, queryVector, ef);
  }

  /**
   * Prints a query to a string, with <code>field</code> assumed to be the
   * default field and omitted.
   *
   * @param field field for serialize to String
   */
  @Override
  public String toString(String field) {
    return String.format(Locale.ROOT, "KnnIvfFlatQuery{field=%s;fromQuery=%s;numCentroids=%d}", field, queryVector, numCentroids);
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
   * @param other
   * @see #sameClassAs(Object)
   * @see #classHash()
   */
  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && equalsTo(getClass().cast(other));
  }

  /**
   * Override and implement query hash code properly in a subclass.
   * This is required so that {@link QueryCache} works properly.
   *
   * @see #equals(Object)
   */
  @Override
  public int hashCode() {
    return classHash() + Objects.hash(field, numCentroids, queryVector);
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
        Objects.equals(queryVector, other.queryVector) &&
        Objects.equals(numCentroids, other.numCentroids);
  }
}
