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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

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

  public IvfFlatIndex setClusteredPoints(List<ClusteredPoints> clusteredPoints) {
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
  SortedImmutableVectorValue search(float[] query, int ef, int centroids, VectorValues vectorValues) throws IOException {
    if (clusteredPoints.isEmpty()) {
      return new SortedImmutableVectorValue(0, query, this.distanceMeasure);
    }

    ensureCentroids(vectorValues);

    /// Phase one -> search top centroids
    final List<ImmutableUnClusterableVector> immutableUnClusterableVectors = new ArrayList<>(clusteredPoints.size());
    clusteredPoints.forEach(i -> immutableUnClusterableVectors.add(new ImmutableUnClusterableVector(
        i.getCenter(), this.distanceMeasure.compute(i.getCentroidValue(), query), i.getPoints())));
    List<ImmutableUnClusterableVector> clusters = immutableUnClusterableVectors;
    if (clusteredPoints.size() > centroids) {
      clusters = immutableUnClusterableVectors.stream().sorted((o1, o2) -> {
            if (o1.distance() < o2.distance()) {
              return -1;
            } else if (o2.distance() < o1.distance()) {
              return 1;
            }

            return 0;
          }).limit(centroids).collect(Collectors.toList());
    }

    SortedImmutableVectorValue results = new SortedImmutableVectorValue(ef, query, this.distanceMeasure);
    clusters.forEach(results::insertWithOverflow);

    /// Phase two -> search topK center points from top centroids and their clusters
    IOException[] exceptions = new IOException[]{null};
    clusters.forEach(cluster -> cluster.points().forEach(docId -> {
      if (docId != cluster.docId()) {
        try {
          if (!vectorValues.seek(docId)) {
            throw new IllegalStateException("docId=" + docId + " has no vector value");
          }

          results.insertWithOverflow(new ImmutableUnClusterableVector(docId, this.distanceMeasure.compute(
              vectorValues.vectorValue(), query)));
        } catch (IOException e) {
          exceptions[0] = e;
        }
      }
    }));

    if (exceptions[0] != null) {
      throw exceptions[0];
    }

    assert results.size() <= ef;

    return results;
  }

  private void ensureCentroids(VectorValues vectorValues) throws IOException {
    for (ClusteredPoints clusteredPoint : clusteredPoints) {
      if (clusteredPoint.getCentroidValue() != null) {
        continue;
      }

      if (!vectorValues.seek(clusteredPoint.getCenter())) {
        throw new IllegalStateException("docId=" + clusteredPoint.getCenter() + " has no vector value");
      }

      clusteredPoint.setCentroidValue(vectorValues.vectorValue().clone());
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
    /// cache vector values for centroid, non-centroid should be null
    private float[] centroidValue;
    private final List<Integer> pointsNearCentroid;

    public ClusteredPoints(int centroid, List<Integer> pointsNearCentroid) {
      this(centroid, null, pointsNearCentroid);
    }

    public ClusteredPoints(int centroid, float[] centroidValue, List<Integer> pointsNearCentroid) {
      this.centroid = centroid;
      this.centroidValue = centroidValue;
      this.pointsNearCentroid = pointsNearCentroid;
    }

    public int getCenter() {
      return this.centroid;
    }

    public void setCentroidValue(float[] centroidValue) {
      this.centroidValue = centroidValue;
    }

    public List<Integer> getPoints() {
      return this.pointsNearCentroid;
    }

    public float[] getCentroidValue() {
      return this.centroidValue;
    }

    static List<ClusteredPoints> convert(List<Centroid<ImmutableClusterableVector>> collection) {
      return collection.stream().map(centroid -> {
        final List<Integer> clusterList = new ArrayList<>(centroid.getPoints().size());
        centroid.getPoints().stream().map(Clusterable::docId).forEach(clusterList::add);
        return new ClusteredPoints(centroid.getCenter().docId(),
            centroid.getCenter().getPoint(), clusterList);

      }).collect(Collectors.toList());
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

  /** Lazy initialization. */
  public static IvfFlatIndex emptyInstance() {
    return IvfFlatIndexHolder.EMPTY_INDEX;
  }
}
