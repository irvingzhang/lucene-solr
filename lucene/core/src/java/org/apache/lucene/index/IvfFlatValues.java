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

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IntsRef;

/**
 * Access to the center points and their invert index link.
 */
public abstract class IvfFlatValues extends DocIdSetIterator {

  /** Sole constructor */
  protected IvfFlatValues() {}

  /**
   * Returns the center points of the ivfflat index.
   * @return centroids
   */
  public abstract int[] getCentroids();

  /**
   * Returns the inverse list (doc ID list) that belongs to the {@code centroid}.
   * Each doc ID in the list is closer to the current {@code centroid} than any other center points.
   * {@code centroid} must be a valid doc ID, ie. &ge; 0 and &le; {@link #NO_MORE_DOCS}.
   * It is illegal to call this method after {@link #advanceExact(int)}
   * returned {@code false}.
   * @param level the graph level
   * @return friend list
   */
  public abstract IntsRef getIvfLink(int centroid);

  /** Move the pointer to exactly {@code target} and return whether
   *  {@code target} has friends lists.
   *  {@code target} must be a valid doc ID, ie. &ge; 0 and &lt; {@code maxDoc}.
   *  After this method returns, {@link #docID()} retuns {@code target}. */
  public abstract boolean advanceExact(int target) throws IOException;

  /** Empty graph value */
  public static IvfFlatValues EMPTY = new IvfFlatValues() {
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
      return NO_MORE_DOCS;
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
      return NO_MORE_DOCS;
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
     * @param target
     * @since 2.9
     */
    @Override
    public int advance(int target) throws IOException {
      return NO_MORE_DOCS;
    }

    /**
     * Returns the estimated cost of this {@link DocIdSetIterator}.
     * <p>
     * This is generally an upper bound of the number of documents this iterator
     * might match, but may be a rough heuristic, hardcoded value, or otherwise
     * completely inaccurate.
     */
    @Override
    public long cost() {
      return 0;
    }

    /**
     * Returns the center points of the ivfflat index.
     *
     * @return centroids
     */
    @Override
    public int[] getCentroids() {
      return new int[0];
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
      return new IntsRef();
    }

    /**
     * Move the pointer to exactly {@code target} and return whether
     * {@code target} has friends lists.
     * {@code target} must be a valid doc ID, ie. &ge; 0 and &lt; {@code maxDoc}.
     * After this method returns, {@link #docID()} retuns {@code target}.
     *
     * @param target
     */
    @Override
    public boolean advanceExact(int target) throws IOException {
      return false;
    }
  };
}
