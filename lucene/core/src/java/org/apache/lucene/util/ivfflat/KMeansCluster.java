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
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;

import org.apache.lucene.index.VectorValues;

/**
 * KMeans clustering
 *
 * TODO Consider training faster
 */
public class KMeansCluster<T extends Clusterable> implements Clusterer<T> {
  /** Max iteration for k-means, a trade-off for efficiency and divergence. */
  private static final int MAX_KMEANS_ITERATIONS = 10;

  /** Default k for k-means clustering. */
  private static final int DEFAULT_KMEANS_K = 1000;

  private int maxIterations;

  private int k;

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
  }

  /** Cluster points on the basis of a similarity measure
   *
   * @param trainingPoints collection of training points.
   */
  @Override
  public List<Centroid<T>> cluster(List<T> trainingPoints) throws NoSuchElementException {
    int collectionSize = trainingPoints.size();
    /// A useful reference: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
    /*if (collectionSize <= 1E6) {
      this.k = Math.min(collectionSize, (int) Math.sqrt(collectionSize) << 2);
    } else if (collectionSize <= 1E7) { /// starting from here, training would be slow
      /// TODO IVF in combination with HNSW uses HNSW to do the cluster assignment
      this.k = 65536;
    } else if (collectionSize <= 1E8) { /// not recommend training on so large data sets
      this.k = 262144;
    } else {
      this.k = 1048576;
    }*/

    int numCentroids = Math.min(collectionSize, (int) Math.sqrt(collectionSize) << 2);
    /// 12.5% or k * 32 points for training
    int trainingSize = Math.min(collectionSize, Math.max(collectionSize >> 3, numCentroids << 5));

    /// shuffle the whole collection
    Collections.shuffle(trainingPoints);

    /// select a subset for training
    final List<T> trainingSubset = trainingPoints.subList(0, trainingSize);

    final List<T> untrainedSubset = trainingPoints.subList(trainingSize, collectionSize);

    /// training
    final List<Centroid<T>> centroidList = cluster(trainingSubset, numCentroids);

    /// insert each untrained point to the nearest cluster
    this.assignPoints(untrainedSubset, centroidList);

    return centroidList;
  }

  /**
   * Cluster points on the basis of a similarity measure
   *
   * @param trainingPoints   collection of training points.
   * @param numCentroids     specify the parameter k for k-means training
   * @return the cluster points with corresponding centroid
   * @throws NoSuchElementException if the clustering not converge
   */
  @Override
  public List<Centroid<T>> cluster(Collection<T> trainingPoints, int numCentroids) throws NoSuchElementException {
    assert numCentroids <= trainingPoints.size();
    this.k = numCentroids;
    final List<Centroid<T>> clusters = this.initCentroids(trainingPoints);
    int max = this.maxIterations < 0 ? MAX_KMEANS_ITERATIONS : this.maxIterations;

    boolean isCenterChanged = true;
    for (int count = 0; count < max; ++count) {
      isCenterChanged = this.clustering(trainingPoints, clusters);
      if (isCenterChanged) {
        clusters.forEach(c -> c.getPoints().clear());
      } else {
        break;
      }
    }

    if (isCenterChanged) {
      this.assignPoints(trainingPoints, clusters);
    }

    return clusters;
  }

  private void assignPoints(final Collection<T> points, final List<Centroid<T>> clusters) {
    for (final T point : points) {
      final Optional<Centroid<T>> nearestCentroid = clusters.stream().min((o1, o2) -> {
        float lhs = distance(o1.getCenter(), point);
        float rhs = distance(o2.getCenter(), point);

        if (lhs < rhs) {
          return -1;
        } else if (rhs < lhs) {
          return 1;
        }

        return 0;
      });

      assert nearestCentroid.isPresent();
      nearestCentroid.get().addPoint(point);
    }
  }

  private List<Centroid<T>> initCentroids(final Collection<T> points) {
    final List<Centroid<T>> centroids = new ArrayList<>(this.k);
    int docId = 0;
    for (Iterator<T> i$ = points.iterator(); i$.hasNext() && centroids.size() < this.k;) {
      centroids.add(new Centroid<>(i$.next().clone().setDocId(docId++)));
    }

    assert centroids.size() == this.k;

    return centroids;
  }

  private boolean clustering(final Collection<T> points, final List<Centroid<T>> centroids) {
    for (final T point : points) {
      int bestCentroid = -1;
      float bestDist = Float.MAX_VALUE;
      for (int i = 0; i < centroids.size(); ++i) {
        float currentDist = distance(point, centroids.get(i).getCenter());
        if (currentDist < bestDist) {
          bestDist = currentDist;
          bestCentroid = i;
        }
      }

      centroids.get(bestCentroid).addPoint(point);
    }

    int dims = centroids.get(0).getCenter().getPoint().length;
    boolean isCenterChanged = false;
    for (int i = 0; i < this.k; ++i) {
      Clusterable newCenter = this.centroidOf(i, centroids.get(i), dims);
      if (!isCenterChanged && !Arrays.equals(centroids.get(i).getCenter().getPoint(), newCenter.getPoint())) {
        isCenterChanged = true;
      }
      centroids.get(i).setCenter(newCenter);
    }

    return isCenterChanged;
  }

  private Clusterable centroidOf(int docId, final Centroid<T> points, int dimension) {
    final float[] centroid = new float[dimension];
    points.getPoints().forEach(p -> {
      final float[] point = p.getPoint();
      for (int i = 0; i < centroid.length; ++i) {
        centroid[i] += point[i];
      }
    });

    float pointSize = (float) points.getPoints().size();
    for (int i = 0; i < centroid.length; ++i) {
      centroid[i] /= pointSize;
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
