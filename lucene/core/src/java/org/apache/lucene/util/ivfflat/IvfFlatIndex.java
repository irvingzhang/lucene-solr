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

import java.io.IOException;
import java.io.StreamCorruptedException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.stream.Collectors;

import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.RamUsageEstimator;

public class IvfFlatIndex implements Accountable {
  private final DistanceMeasure distanceMeasure;
  private List<ClusteredPoints> clusteredPoints;

  private long ramBytesUsed;

  public IvfFlatIndex(VectorValues.DistanceFunction distFunc) {
    this.distanceMeasure = DistanceFactory.instance(distFunc);
    this.clusteredPoints = Collections.EMPTY_LIST;
    this.ramBytesUsed = 0;
  }

  public IvfFlatIndex setClusteredPoints(final List<ClusteredPoints> clusteredPoints) {
    this.clusteredPoints = clusteredPoints;
    return this;
  }

  public IvfFlatIndex finish() {
    return this;
  }

  /**
   * Searches the nearest neighbors for a specified query.
   *
   * @param query        search query vector
   * @param ef           the number of nodes to be searched
   * @param centroids    max searched centroids
   * @param vectorValues vector values
   * @return sorted results
   */
  SortedImmutableVectorValue search(float[] query, int ef, int centroids, VectorValues vectorValues,
                                    IvfFlatValues ivfFlatValues) throws IOException {
    if (clusteredPoints.isEmpty() || ivfFlatValues == null) {
      return new SortedImmutableVectorValue(0, query, this.distanceMeasure);
    }

    final VectorValues values = ivfFlatValues.getCentroids();

    /// Phase one -> search top centroids
    final List<ImmutableUnClusterableVector> immutableUnClusterableVectors = new ArrayList<>(clusteredPoints.size());
    for (ClusteredPoints clusteredPoint : clusteredPoints) {
      immutableUnClusterableVectors.add(new ImmutableUnClusterableVector(
          clusteredPoint.getCenter(), this.distanceMeasure.compute(clusteredPoint.getCentroidValue(values), query),
          clusteredPoint.getPoints()));
    }

    final List<ImmutableUnClusterableVector> clusters = immutableUnClusterableVectors.stream().sorted((o1, o2) -> {
      if (o1.distance() < o2.distance()) {
        return -1;
      } else if (o2.distance() < o1.distance()) {
        return 1;
      }

      return 0;
    }).collect(Collectors.toList());

    /// if there aren't enough results in nearest centroids, try to search from the candidateClusters
    List<ImmutableUnClusterableVector> expectedClusters, candidateClusters = Collections.EMPTY_LIST;
    if (clusters.size() > centroids) {
      expectedClusters = clusters.subList(0, centroids);
      candidateClusters = clusters.subList(centroids, clusters.size());
    } else {
      expectedClusters = clusters;
    }

    SortedImmutableVectorValue results = new SortedImmutableVectorValue(ef, query, this.distanceMeasure);

    /// Phase two -> search topK center points from top centroids and their clusters
    searchNearestPoints(query, vectorValues, expectedClusters, results, () -> false);

    /// Phase three -> search from candidate clusters to ensure sufficient results
    if (results.size() < ef) {
      searchNearestPoints(query, vectorValues, candidateClusters, results, () -> results.size() >= ef);
    }

    return results;
  }

  private void searchNearestPoints(float[] query, VectorValues vectorValues, List<ImmutableUnClusterableVector> clusters,
                                   SortedImmutableVectorValue results, BooleanSupplier supplier) throws IOException {
    for (ImmutableUnClusterableVector cluster : clusters) {
      final List<Integer> points = cluster.points();
      for (Integer point : points) {
        if (!vectorValues.seek(point)) {
          throw new IllegalStateException("docId=" + point + " has no vector value");
        }

        results.insertWithOverflow(new ImmutableUnClusterableVector(point, this.distanceMeasure.compute(
            vectorValues.vectorValue(), query)));
      }

      if (supplier.getAsBoolean()) {
        break;
      }
    }
  }

  /**
   * Return the memory usage of this object in bytes. Negative values are illegal.
   */
  @Override
  public long ramBytesUsed() {
    if (ramBytesUsed == 0) {
      ramBytesUsed = RamUsageEstimator.sizeOfCollection(this.clusteredPoints);
    }
    return ramBytesUsed;
  }

  public static class ClusteredPoints implements Accountable {
    private final int centroid;
    private final List<Integer> pointsNearCentroid;

    public ClusteredPoints(int centroid, List<Integer> pointsNearCentroid) {
      this.centroid = centroid;
      this.pointsNearCentroid = pointsNearCentroid;
    }

    public int getCenter() {
      return this.centroid;
    }

    public List<Integer> getPoints() {
      return this.pointsNearCentroid;
    }

    public float[] getCentroidValue(VectorValues reader) throws IOException {
      if (reader.seek(centroid)) {
        return reader.vectorValue();
      }

      throw new StreamCorruptedException("Cannot find centroid with id [" + centroid + "]");
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
      return RamUsageEstimator.shallowSizeOfInstance(getClass()) + RamUsageEstimator.sizeOfCollection(pointsNearCentroid);
    }
  }

  private static final class IvfFlatIndexHolder {
    public static final IvfFlatIndex EMPTY_INDEX = new IvfFlatIndex(VectorValues.DistanceFunction.NONE);
  }

  /**
   * Lazy initialization.
   */
  public static IvfFlatIndex emptyInstance() {
    return IvfFlatIndexHolder.EMPTY_INDEX;
  }
}
