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

import java.util.ArrayList;
import java.util.List;

/**
 * {@code Cluster} describes a set of points that belongs to a certain centroid.
 */
public class Cluster<T extends Clusterable> {
  private final ArrayList<T> points = new ArrayList<>();

  /** Add a point to this cluster. */
  public void addPoint(T point) {
    this.points.add(point);
  }

  /** Returns all points belong to this cluster. */
  public List<T> getPoints() {
    return this.points;
  }
}