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

import org.apache.lucene.index.VectorValues;

/**
 * {@code DistanceFactory} is a factory for creating distance measures.
 */
public final class DistanceFactory {

  /** Returns distance measure according to the distance function. */
  public static DistanceMeasure instance(VectorValues.DistanceFunction distFunc) {
    switch (distFunc) {
      case MANHATTAN:
        return (v1, v2) -> VectorValues.DistanceFunction.MANHATTAN.distance(v1, v2);

      case EUCLIDEAN:
        return (v1, v2) -> VectorValues.DistanceFunction.EUCLIDEAN.distance(v1, v2);

      case COSINE:
        return (v1, v2) -> VectorValues.DistanceFunction.COSINE.distance(v1, v2);

      case NONE:
        return (v1, v2) -> 0.0F;

      default:
        throw new UnsupportedOperationException("unsupported distance measure type: " + distFunc);
    }
  }
}
