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

package org.apache.lucene.util.ivfflat;

import org.apache.lucene.util.PriorityQueue;

public class SortedImmutableVectorValue extends PriorityQueue<ImmutableUnClusterableVector> {
  final float[] queryVector;

  final DistanceMeasure distanceMeasure;

  SortedImmutableVectorValue(int maxSize, float[] queryVector, DistanceMeasure distanceMeasure) {
    super(maxSize);
    this.queryVector = queryVector;
    this.distanceMeasure = distanceMeasure;
  }
  /**
   * Determines the ordering of objects in this priority queue.  Subclasses
   * must define this one method.
   *
   * @param a
   * @param b
   * @return <code>true</code> iff parameter <tt>a</tt> is less than parameter <tt>b</tt>.
   */
  @Override
  protected boolean lessThan(ImmutableUnClusterableVector a, ImmutableUnClusterableVector b) {
    return a.distance() < b.distance();
  }
}
