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
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.LuceneTestCase;

public class TestKmeansCluster extends LuceneTestCase {
  private Random random = new Random();

  private static final int dims = 256;

  private static final int dataSize = 40000;

  public void testCluster() {
    final List<float[]> vectors = randomVectors();
    assertEquals(dataSize, vectors.size());

    final List<ImmutableClusterableVector> clusterableVectors = new ArrayList<>(dataSize);
    int counter = 0;
    for (float[] vector : vectors) {
      clusterableVectors.add(new ImmutableClusterableVector(counter++, vector));
    }

    final KMeansCluster<ImmutableClusterableVector> cluster = new KMeansCluster<>(VectorValues.DistanceFunction.EUCLIDEAN);

    long startTime = System.currentTimeMillis();
    final List<Centroid<ImmutableClusterableVector>> centroids = cluster.cluster(clusterableVectors);
    long costTime = System.currentTimeMillis() - startTime;

    /*System.out.println("Total points -> " + clusterableVectors.size() + ", dimension -> " + dims + ", clustering cost -> " +
        costTime + " msec, centroid size -> " + cluster.getK());*/

    assertEquals(cluster.getK(), centroids.size());

    int totalCnts = 0;
    for (Centroid<ImmutableClusterableVector> centroid : centroids) {
      totalCnts += centroid.getPoints().size();
      for (ImmutableClusterableVector vector : centroid.getPoints()) {

        List<Centroid<ImmutableClusterableVector>> results = centroids.stream().sorted((o1, o2) -> {
          double left = cluster.distance(o1.getCenter(), vector);
          double right = cluster.distance(o2.getCenter(), vector);

          if (left < right) {
            return -1;
          } else if (right < left) {
            return 1;
          }

          return 0;
        }).limit(1).collect(Collectors.toList());

        assertEquals(1, results.size());

        assertEquals(centroid.getCenter().docId(), results.get(0).getCenter().docId());

        assertEquals(0, Arrays.compare(centroid.getCenter().getPoint(), results.get(0).getCenter().getPoint()));
      }
    }

    assertEquals(dataSize, totalCnts);
  }

  private List<float[]> randomVectors() {
    List<float[]> vectors = new ArrayList<>(TestKmeansCluster.dataSize);
    for (int i = 0; i < TestKmeansCluster.dataSize; ++i) {
      vectors.add(randomVector());
    }

    return vectors;
  }

  private float[] randomVector() {
    float[] vector = new float[TestKmeansCluster.dims];
    for(int i = 0; i < TestKmeansCluster.dims; i++) {
      vector[i] = random.nextFloat();
    }

    return vector;
  }
}
