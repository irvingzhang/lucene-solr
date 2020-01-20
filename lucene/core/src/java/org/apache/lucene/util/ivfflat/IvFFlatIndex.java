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

public class IvFFlatIndex implements Accountable {
  private final DistanceMeasure distanceMeasure;
  private List<ClusteredPoints> clusteredPoints;

  private boolean frozen = false;
  private long ramBytesUsed;

  public IvFFlatIndex(VectorValues.DistanceFunction distFunc) {
    this.distanceMeasure = DistanceFactory.instance(distFunc);
    this.clusteredPoints = Collections.EMPTY_LIST;
    this.ramBytesUsed = 0;
  }

  public IvFFlatIndex setClusteredPoints(List<ClusteredPoints> clusteredPoints) {
    this.clusteredPoints = clusteredPoints;
    return this;
  }

  public IvFFlatIndex finish() {
    this.frozen = true;
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
    ensureCentroids(vectorValues);

    /// Phase one -> search top center points and their invert index links
    final List<ImmutableUnClusterableVector> immutableUnClusterableVectors = new ArrayList<>(clusteredPoints.size());
    clusteredPoints.forEach(i -> immutableUnClusterableVectors.add(new ImmutableUnClusterableVector(
        i.getCenter(), this.distanceMeasure.compute(i.getCentroidValue(), query), i.getPoints())));
    final List<ImmutableUnClusterableVector> clusters = immutableUnClusterableVectors.stream()
        .sorted((o1, o2) -> {
          if (o1.distance() < o2.distance()) {
            return -1;
          } else if (o2.distance() < o1.distance()) {
            return 1;
          }

          return 0;
        }).limit(centroids).collect(Collectors.toList());

    SortedImmutableVectorValue results = new SortedImmutableVectorValue(ef, query, this.distanceMeasure);
    clusters.forEach(cluster -> results.insertWithOverflow(cluster));

    /// Phase two -> search topK center points and their inverted index links
    IOException[] exceptions = new IOException[]{null};
    clusters.forEach(cluster -> cluster.points().forEach(docId -> {
      if (docId != cluster.docId()) {
        try {
          if (!vectorValues.seek(docId)) {
            throw new IllegalStateException("docId=" + docId + " has no vector value");
          }

          results.add(new ImmutableUnClusterableVector(docId, this.distanceMeasure.compute(
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
    IOException[] exceptions = new IOException[]{null};
    clusteredPoints.forEach(point -> {
      if (point.getCentroidValue() == ClusteredPoints.EMPTY_FLOAT) {
        try {
          if (!vectorValues.seek(point.getCenter())) {
            throw new IllegalStateException("docId=" + point.getCenter() + " has no vector value");
          }

          point.centroidValue = vectorValues.vectorValue();
        } catch (IOException e) {
          exceptions[0] = e;
        }
      }
    });

    if (exceptions[0] != null) {
      throw exceptions[0];
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
    final static float[] EMPTY_FLOAT = new float[0];
    final int centroid;
    /// TODO cache vector values for center points
    float[] centroidValue;
    final List<Integer> pointsNearCentroid;

    public ClusteredPoints(int centroid, List<Integer> pointsNearCentroid) {
      this(centroid, EMPTY_FLOAT, pointsNearCentroid);
    }

    public ClusteredPoints(int centroid, float[] centroidValue, List<Integer> pointsNearCentroid) {
      this.centroid = centroid;
      this.centroidValue = centroidValue;
      this.pointsNearCentroid = pointsNearCentroid;
    }

    public int getCenter() {
      return this.centroid;
    }

    public List<Integer> getPoints() {
      return this.pointsNearCentroid;
    }

    public float[] getCentroidValue() {
      return this.centroidValue;
    }

    static List<ClusteredPoints> convert(List<Centroid<ImmutableClusterableVector>> collection) {
      List<ClusteredPoints> results = new ArrayList<>(collection.size());
      collection.forEach(clusterableCentroid -> {
        List<Integer> clusterList = new ArrayList<>(clusterableCentroid.getPoints().size());
        clusterableCentroid.getPoints().forEach(point -> clusterList.add(point.docId()));
        results.add(new ClusteredPoints(clusterableCentroid.getCenter().docId(),
            clusterableCentroid.getCenter().getPoint(), clusterList));
      });

      return results;
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
      return RamUsageEstimator.shallowSizeOfInstance(getClass()) + RamUsageEstimator.sizeOfCollection(pointsNearCentroid);
    }
  }
}
