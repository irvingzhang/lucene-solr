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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.IntsRef;

public final class IvfFlatCacheReader {
  private static final Map<IvfFlatCacheKey, IvFFlatIndex> IVF_FLAT_INDEX_CACHE = new ConcurrentHashMap<>();

  private final String field;
  private final LeafReaderContext context;

  public IvfFlatCacheReader(String field, LeafReaderContext context) {
    this.field = field;
    this.context = context;
  }

  public SortedImmutableVectorValue search(float[] query, int ef, int numCentroids, VectorValues vectorValues) throws IOException {
    IvFFlatIndex ivFFlatIndex = get(field, context, false);
    return ivFFlatIndex.search(query, ef, numCentroids, vectorValues);
  }

  public static long loadIvfFlats(String field, IndexReader reader, boolean forceReload) throws IOException {
    long bytesUsed = 0L;
    for (LeafReaderContext ctx : reader.leaves()) {
      IvFFlatIndex ivFFlatIndex = get(field, ctx, forceReload);
      bytesUsed += ivFFlatIndex.ramBytesUsed();
    }
    return bytesUsed;
  }

  private static IvFFlatIndex get(String field, LeafReaderContext context, boolean forceReload) throws IOException {
    IvfFlatCacheKey key = new IvfFlatCacheKey(field, context.id());
    IOException[] exc = new IOException[]{null};
    if (forceReload) {
      IVF_FLAT_INDEX_CACHE.put(key, load(field, context));
    } else {
      IVF_FLAT_INDEX_CACHE.computeIfAbsent(key, (k -> {
        try {
          return load(k.fieldName, context);
        } catch (IOException e) {
          exc[0] = e;
          return null;
        }
      }));
      if (exc[0] != null) {
        throw exc[0];
      }
    }
    return IVF_FLAT_INDEX_CACHE.get(key);
  }

  private static IvFFlatIndex load(String field, LeafReaderContext context) throws IOException {
    FieldInfo fi = context.reader().getFieldInfos().fieldInfo(field);
    int numDimensions = fi.getVectorNumDimensions();
    if (numDimensions == 0) {
      // the field has no vector values
      return null;
    }
    VectorValues.DistanceFunction distFunc = fi.getVectorDistFunc();

    IvfFlatValues ivfFlatValues = context.reader().getIvfFlatValues(field);
    return load(distFunc, ivfFlatValues);
  }

  public static IvFFlatIndex load(VectorValues.DistanceFunction distFunc, IvfFlatValues ivfFlatValues) throws IOException {
    IvFFlatIndex ivFFlatIndex = new IvFFlatIndex(distFunc);
    int[] centroids = ivfFlatValues.getCentroids();
    List<IvFFlatIndex.ClusteredPoints> clusteredPointsList = new ArrayList<>(centroids.length);
    for (int centroid : centroids) {
      IntsRef ivfLink = ivfFlatValues.getIvfLink(centroid);
      IvFFlatIndex.ClusteredPoints clusteredPoints = new IvFFlatIndex.ClusteredPoints(centroid,
          Arrays.stream(ivfLink.ints).boxed().collect(Collectors.toList()));

      clusteredPointsList.add(clusteredPoints);
    }

    return ivFFlatIndex.setClusteredPoints(clusteredPointsList).finish();
  }

  static final class IvfFlatCacheKey {
    final String fieldName;

    final Object readerId;

    IvfFlatCacheKey(String fieldName, Object readerId) {
      this.fieldName = fieldName;
      this.readerId = readerId;
    }
  }
}
