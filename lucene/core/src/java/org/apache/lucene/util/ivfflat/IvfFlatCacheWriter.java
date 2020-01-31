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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;

import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.RamUsageEstimator;

public final class IvfFlatCacheWriter implements Accountable {
  private final IntsRefBuilder docsRef;
  private final List<float[]> rawVectors;
  private final int numDimensions;
  private final IvfFlatIndex ivFFlatIndex;
  private final KMeansCluster<ImmutableClusterableVector> clusterer;

  private int addedDocs = 0;
  private int lastDocID = -1;

  public IvfFlatCacheWriter(int numDimensions, VectorValues.DistanceFunction distFunc) {
    this.numDimensions = numDimensions;
    this.docsRef = new IntsRefBuilder();
    this.rawVectors = new ArrayList<>();
    this.ivFFlatIndex = new IvfFlatIndex(distFunc);
    this.clusterer = new KMeansCluster<>(distFunc);
  }

  /** Inserts a doc with vector value to the graph */
  public void insert(int docId, BytesRef binaryValue) {
    // add the vector value
    float[] value = VectorValues.decode(binaryValue, numDimensions);
    rawVectors.add(value);
    docsRef.grow(docId + 1);
    docsRef.setIntAt(docId, addedDocs++);
    docsRef.setLength(docId + 1);
    if (docId > lastDocID + 1) {
      Arrays.fill(docsRef.ints(), lastDocID + 1, docId, -1);
    }
    lastDocID = docId;
  }

  public void finish() {
    ivFFlatIndex.finish();
  }

  public List<IvfFlatIndex.ClusteredPoints> cluster(List<ImmutableClusterableVector>
                                                        immutableClusterableVectors) throws NoSuchElementException {
    /// to accelerate training on large data set, select partial points after shuffling for k-means clustering
    if (immutableClusterableVectors.size() > KMeansCluster.MAX_ALLOW_TRAINING_POINTS) {
      /// shuffle the whole collection
      Collections.shuffle(immutableClusterableVectors);

      /// select a subset for training
      final List<ImmutableClusterableVector> trainingSubset = immutableClusterableVectors.subList(0,
          KMeansCluster.MAX_ALLOW_TRAINING_POINTS);

      final List<ImmutableClusterableVector> untrainedSubset = immutableClusterableVectors.subList(
          KMeansCluster.MAX_ALLOW_TRAINING_POINTS, immutableClusterableVectors.size());

      /// training
      final List<Centroid<ImmutableClusterableVector>> centroidList = clusterer.cluster(trainingSubset,
          (int) Math.sqrt(immutableClusterableVectors.size() >> 1));

      /// insert each untrained point to the nearest cluster
      untrainedSubset.forEach(point -> {
        final Optional<Centroid<ImmutableClusterableVector>> nearestCentroid = centroidList.stream().min((o1, o2) -> {
          float lhs = clusterer.getDistanceMeasure().compute(o1.getCenter().getPoint(), point.getPoint());
          float rhs = clusterer.getDistanceMeasure().compute(o2.getCenter().getPoint(), point.getPoint());

          if (lhs < rhs) {
            return -1;
          } else if (rhs < lhs) {
            return 1;
          }

          return 0;
        });

        assert nearestCentroid.isPresent();
        nearestCentroid.get().addPoint(point);
      });

      return IvfFlatIndex.ClusteredPoints.convert(centroidList);
    } else {
      return IvfFlatIndex.ClusteredPoints.convert(clusterer.cluster(immutableClusterableVectors));
    }
  }

  public float[][] rawVectorsArray() {
    return rawVectors.toArray(new float[0][]);
  }

  @Override
  public long ramBytesUsed() {
    // calculating the exact ram usage is time consuming so we make rough estimation here
    return RamUsageEstimator.sizeOf(docsRef.ints()) +
        Float.BYTES * numDimensions * rawVectors.size() +
        RamUsageEstimator.shallowSizeOfInstance(IvfFlatIndex.class);
  }
}
