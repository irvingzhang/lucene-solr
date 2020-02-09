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
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import org.apache.lucene.index.VectorValues;

/**
 * Migrate from {@link org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer}
 * with minor refactoring, avoiding to introduce external dependencies.
 */
public class KMeansCluster<T extends Clusterable> implements Clusterer<T> {
  /** Maximum allowed point number for training, avoid training last too long */
  public static final int MAX_ALLOW_TRAINING_POINTS = 200000;

  /** Max iteration for k-means, a trade-off for efficiency and divergence. */
  private static final int MAX_KMEANS_ITERATIONS = 10;

  /** Default k for k-means clustering. */
  private static final int DEFAULT_KMEANS_K = 1000;

  private int maxIterations;

  private int k;

  private final Random random;

  private final DistanceMeasure distanceMeasure;

  public KMeansCluster(VectorValues.DistanceFunction distFunc) {
    this(MAX_KMEANS_ITERATIONS, DEFAULT_KMEANS_K, distFunc);
  }

  public KMeansCluster(int k, VectorValues.DistanceFunction distFunc) {
    this(MAX_KMEANS_ITERATIONS, k, distFunc);
  }

  public KMeansCluster(int maxIterations, int k, VectorValues.DistanceFunction distFunc) {
    this.maxIterations = maxIterations;
    this.k = k;
    this.distanceMeasure = DistanceFactory.instance(distFunc);
    this.random = new Random();
  }

  /** Cluster points on the basis of a similarity measure
   *
   * @param trainingPoints collection of training points.
   */
  @Override
  public List<Centroid<T>> cluster(Collection<T> trainingPoints) throws NoSuchElementException {
    /// adaptively choose the value for k
    return this.cluster(trainingPoints, (int) Math.sqrt(trainingPoints.size()) << 2);
  }

  /**
   * Cluster points on the basis of a similarity measure
   *
   * @param trainingPoints collection of training points.
   * @param expectK        specify the parameter for k-means training
   * @return the cluster points with corresponding centroid
   * @throws NoSuchElementException if the clustering not converge
   */
  @Override
  public List<Centroid<T>> cluster(Collection<T> trainingPoints, int expectK) throws NoSuchElementException {
    assert !trainingPoints.isEmpty();

    this.k = Math.min(trainingPoints.size(), expectK);

    List<Centroid<T>> clusters = this.initCenters(trainingPoints);
    int[] assignments = new int[trainingPoints.size()];
    this.assignPointsToClusters(clusters, trainingPoints, assignments);
    int max = this.maxIterations < 0 ? MAX_KMEANS_ITERATIONS : this.maxIterations;

    for (int count = 0; count < max; ++count) {
      boolean emptyCluster = false;
      final List<Centroid<T>> newClusters = new ArrayList<>();

      Clusterable newCenter;
      for (Iterator<Centroid<T>> i$ = clusters.iterator(); i$.hasNext();
           newClusters.add(new Centroid<>(newCenter))) {
        final Centroid<T> cluster = i$.next();
        if (cluster.getPoints().isEmpty()) {
          newCenter = this.getFarthestPoint(clusters);
          emptyCluster = true;
        } else {
          newCenter = this.centroidOf(cluster.getCenter().docId(), cluster.getPoints(),
              cluster.getCenter().getPoint().length);
        }
      }

      int changes = this.assignPointsToClusters(newClusters, trainingPoints, assignments);
      clusters = newClusters;
      if (changes == 0 && !emptyCluster) {
        return newClusters;
      }
    }

    return clusters;
  }

  private List<Centroid<T>> initCenters(final Collection<T> points) {
    final List<T> pointList = List.copyOf(points);
    int numPoints = pointList.size();
    boolean[] visited = new boolean[numPoints];
    final List<Centroid<T>> resultSet = new ArrayList<>(this.k);

    /// random select the first point
    int firstPointIndex = this.random.nextInt(numPoints);
    T firstPoint = pointList.get(firstPointIndex);
    resultSet.add(new Centroid<>(firstPoint));
    visited[firstPointIndex] = true;
    float[] minDistSquared = new float[numPoints];

    for (int i = 0; i < numPoints; ++i) {
      if (i != firstPointIndex) {
        float d = this.distance(firstPoint, pointList.get(i));
        minDistSquared[i] = d * d;
      }
    }

    while (resultSet.size() < this.k) {
      float distSqSum = 0.0F;

      for (int i = 0; i < numPoints; ++i) {
        if (!visited[i]) {
          distSqSum += minDistSquared[i];
        }
      }

      float r = this.random.nextFloat() * distSqSum;
      int nextPointIndex = -1;
      float sum = 0.0F;

      int i = 0;
      for (;i < numPoints; ++i) {
        if (!visited[i]) {
          sum += minDistSquared[i];
          if (sum >= r) {
            nextPointIndex = i;
            break;
          }
        }
      }

      if (nextPointIndex == -1) {
        for (i = numPoints - 1; i >= 0; --i) {
          if (!visited[i]) {
            nextPointIndex = i;
            break;
          }
        }
      }

      if (nextPointIndex < 0) {
        break;
      }

      T p = pointList.get(nextPointIndex);
      resultSet.add(new Centroid<>(p));
      visited[nextPointIndex] = true;
      if (resultSet.size() < this.k) {
        for (int j = 0; j < numPoints; ++j) {
          if (!visited[j]) {
            float d = this.distance(p, pointList.get(j));
            float d2 = d * d;
            if (d2 < minDistSquared[j]) {
              minDistSquared[j] = d2;
            }
          }
        }
      }
    }

    return resultSet;
  }

  private int assignPointsToClusters(final List<Centroid<T>> clusters, final Collection<T> points, int[] assignments) {
    int assignedDifferently = 0;
    int pointIndex = 0;

    int clusterIndex;
    for (Iterator<T> i$ = points.iterator(); i$.hasNext(); assignments[pointIndex++] = clusterIndex) {
      T p = i$.next();
      clusterIndex = this.getNearestCluster(clusters, p);
      if (clusterIndex != assignments[pointIndex]) {
        ++assignedDifferently;
      }

      final Centroid<T> cluster = clusters.get(clusterIndex);
      cluster.addPoint(p);
    }

    return assignedDifferently;
  }

  private int getNearestCluster(final Collection<Centroid<T>> clusters, T point) {
    float minDistance = Float.MAX_VALUE;
    int clusterIndex = 0, minCluster = 0;

    for (Iterator<Centroid<T>> i$ = clusters.iterator(); i$.hasNext(); ++clusterIndex) {
      float distance = this.distance(point, i$.next().getCenter());
      if (distance < minDistance) {
        minDistance = distance;
        minCluster = clusterIndex;
      }
    }

    return minCluster;
  }

  private T getFarthestPoint(final Collection<Centroid<T>> clusters) throws NoSuchElementException {
    float maxDistance = -1.0F / 0.0F;
    Cluster<T> selectedCluster = null;
    int selectedPoint = -1;

    for (Centroid<T> cluster : clusters) {
      final Clusterable center = cluster.getCenter();
      final List<T> points = cluster.getPoints();

      for (int i = 0; i < points.size(); ++i) {
        float distance = this.distance(points.get(i), center);
        if (distance > maxDistance) {
          maxDistance = distance;
          selectedCluster = cluster;
          selectedPoint = i;
        }
      }
    }

    if (selectedCluster == null) {
      throw new NoSuchElementException("Cannot find point from largest number cluster");
    } else {
      return selectedCluster.getPoints().remove(selectedPoint);
    }
  }

  private Clusterable centroidOf(int docId, final Collection<T> points, int dimension) {
    float[] centroid = new float[dimension];

    for (T p : points) {
      float[] point = p.getPoint();

      for (int i = 0; i < centroid.length; ++i) {
        centroid[i] += point[i];
      }
    }

    for (int i = 0; i < centroid.length; ++i) {
      centroid[i] /= (float) points.size();
    }

    return new ImmutableClusterableVector(docId, centroid);
  }

  @Override
  public float distance(Clusterable p1, Clusterable p2) {
    return this.distanceMeasure.compute(p1.getPoint(), p2.getPoint());
  }

  public int getK() {
    return k;
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }
}
