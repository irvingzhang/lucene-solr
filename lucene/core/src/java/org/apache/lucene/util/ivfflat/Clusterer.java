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

import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.lucene.util.ivfflat.Cluster;
import org.apache.lucene.util.ivfflat.Clusterable;

/**
 * {@code Clusterer} is a executor for clustering.
 */
public interface Clusterer<T extends Clusterable> {
    /** Cluster points on the basis of a similarity measure
     *
     * @param trainingPoints collection of training points.
     *
     * @throws NoSuchElementException
     */
    List<? extends Cluster<T>> cluster(Collection<T> trainingPoints) throws NoSuchElementException;

    /**
     * Cluster points on the basis of a similarity measure
     *
     * @param trainingPoints collection of training points.
     * @param expectK specify the parameter for k-means training
     * @return
     * @throws NoSuchElementException
     */
    List<? extends Cluster<T>> cluster(Collection<T> trainingPoints, int expectK) throws NoSuchElementException;

    /** Distance by some measure means **/
    float distance(Clusterable p1, Clusterable p2);
}
