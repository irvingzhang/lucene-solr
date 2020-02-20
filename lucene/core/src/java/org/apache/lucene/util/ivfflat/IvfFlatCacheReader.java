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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.IntsRef;

public final class IvfFlatCacheReader {
  private final String field;
  private final LeafReaderContext context;

  public IvfFlatCacheReader(String field, LeafReaderContext context) {
    this.field = field;
    this.context = context;
  }

  public SortedImmutableVectorValue search(float[] query, int ef, int numCentroids,
                                           VectorValues vectorValues) throws IOException {
    return load(field, context).search(query, ef, numCentroids, vectorValues);
  }

  public static long loadIvfFlats(String field, IndexReader reader) throws IOException {
    long bytesUsed = 0L;
    for (LeafReaderContext ctx : reader.leaves()) {
      final IvfFlatIndex ivFFlatIndex = load(field, ctx);
      bytesUsed += ivFFlatIndex.ramBytesUsed();
    }
    return bytesUsed;
  }

  private static IvfFlatIndex load(String field, LeafReaderContext context) throws IOException {
    final FieldInfo fi = context.reader().getFieldInfos().fieldInfo(field);
    if (fi.getVectorNumDimensions() == 0) {
      // the field has no vector values
      return IvfFlatIndex.emptyInstance();
    }

    return load(fi.getVectorDistFunc(), context.reader().getIvfFlatValues(field));
  }

  public static IvfFlatIndex load(VectorValues.DistanceFunction distFunc, final IvfFlatValues ivfFlatValues) throws IOException {
    final int[] centroids = ivfFlatValues.getCentroids();
    final List<IvfFlatIndex.ClusteredPoints> clusteredPointsList = new ArrayList<>(centroids.length);
    for (int centroid : centroids) {
      final IntsRef ivfLink = ivfFlatValues.getIvfLink(centroid);
      IvfFlatIndex.ClusteredPoints clusteredPoints = new IvfFlatIndex.ClusteredPoints(centroid,
          Arrays.stream(ivfLink.ints).boxed().collect(Collectors.toList()));

      clusteredPointsList.add(clusteredPoints);
    }

    return new IvfFlatIndex(distFunc).setClusteredPoints(clusteredPointsList).finish();
  }
}
