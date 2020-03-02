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
import java.util.NoSuchElementException;

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
    final List<Centroid<ImmutableClusterableVector>> clusteredPoints = clusterer.cluster(immutableClusterableVectors);
    assert !clusteredPoints.isEmpty();

    return IvfFlatIndex.ClusteredPoints.convert(clusteredPoints);
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
