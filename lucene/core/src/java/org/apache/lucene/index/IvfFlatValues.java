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

import org.apache.lucene.util.IntsRef;

/**
 * Access to the center points and their invert index link.
 */
public abstract class IvfFlatValues {

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

  /** Empty graph value */
  public static IvfFlatValues EMPTY = new IvfFlatValues() {
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
  };
}
