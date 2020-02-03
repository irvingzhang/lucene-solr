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
package org.apache.lucene.util;


import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.KnnGraphValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnGraphQuery;
import org.apache.lucene.search.KnnIvfFlatQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.junit.Before;

import static org.apache.lucene.util.hnsw.HNSWGraphWriter.RAND_SEED;

/** A simple performance comparison for IVFFlat and HNSW */
public class TestKnnGraphAndIvfFlat extends LuceneTestCase {

  private static final String KNN_VECTOR_FIELD = "vector";

  @Before
  public void setup() {
    RAND_SEED = random().nextLong();
  }

  public void testComparison() throws Exception {
    int numDocs = 2000, dimension = 100;

    float[][] randomVectors = KnnTestHelper.randomVectors(numDocs, dimension);

    runCase(numDocs, dimension, randomVectors, VectorValues.VectorIndexType.IVFFLAT);

    runCase(numDocs, dimension, randomVectors, VectorValues.VectorIndexType.HNSW);
  }

  public static void runCase(int numDoc, int dimension, float[][] randomVectors,
                             VectorValues.VectorIndexType type) throws Exception {
    try (Directory dir = newDirectory(); IndexWriter iw = new IndexWriter(
        dir, newIndexWriterConfig(null).setCodec(Codec.forName("Lucene90")))) {
      for (int i = 0; i < numDoc; ++i) {
        KnnTestHelper.add(iw, i, randomVectors[i], type);
      }

      iw.commit();
      KnnTestHelper.assertConsistent(iw, Arrays.asList(randomVectors), type);

      long totalCostTime = 0;
      int totalRecallCnt = 0;
      QueryResult result;
      int testRecall = Math.min(numDoc, 2000);
      for (int i = 0; i < testRecall; ++i) {
        result = KnnTestHelper.assertRecall(dir, 1, 1, randomVectors[i], false, type);
        totalCostTime += result.costTime;
        totalRecallCnt += result.recallCnt;
      }

      System.out.println("[***" + type.toString() + "***] Total number of docs -> " + numDoc +
          ", dimension -> " + dimension + ", recall experiments -> " + testRecall +
          ", exact recall times -> " + totalRecallCnt + ", total search time -> " +
          totalCostTime + "msec, avg search time -> " + 1.0F * totalCostTime / testRecall +
          "msec, recall percent -> " + 100.0F * totalRecallCnt / testRecall + "%");
    }
  }

  public static final class KnnTestHelper {
    private KnnTestHelper() {
    }

    public static void assertConsistent(IndexWriter iw, List<float[]> values, VectorValues.VectorIndexType type) throws IOException {
      try (DirectoryReader dr = DirectoryReader.open(iw)) {
        KnnGraphValues graphValues;
        IvfFlatValues ivfFlatValues;
        for (LeafReaderContext ctx: dr.leaves()) {
          LeafReader reader = ctx.reader();
          VectorValues vectorValues = reader.getVectorValues(KNN_VECTOR_FIELD);
          if (type == VectorValues.VectorIndexType.HNSW) {
            graphValues = reader.getKnnGraphValues(KNN_VECTOR_FIELD);
            assertEquals((vectorValues == null), (graphValues == null));
          } else if (type == VectorValues.VectorIndexType.IVFFLAT) {
            ivfFlatValues = reader.getIvfFlatValues(KNN_VECTOR_FIELD);
            assertEquals((vectorValues == null), (ivfFlatValues == null));
          }
          if (vectorValues == null) {
            continue;
          }

          for (int i = 0; i < reader.maxDoc(); i++) {
            int id = Integer.parseInt(Objects.requireNonNull(reader.document(i).get("id")));
            if (id < values.size() && values.get(id) == null) {
              // documents without IvfFlatValues have no vectors or neighbors
              assertNotEquals("document " + id + " was not expected to have values", i, vectorValues.advance(i));
            }
          }
        }
      }
    }

    public static void add(IndexWriter iw, int id, float[] vector, VectorValues.VectorIndexType type) throws IOException {
      Document doc = new Document();
      if (vector != null) {
        doc.add(new VectorField(KNN_VECTOR_FIELD, vector, VectorValues.DistanceFunction.EUCLIDEAN, type));
      }
      doc.add(new StringField("id", Integer.toString(id), Field.Store.YES));
      iw.addDocument(doc);
    }

    public static float[][] randomVectors(int numDocs, int numDims) {
      float[][] vectors = new float[numDocs][];
      for (int i = 0; i < numDocs; ++i) {
        vectors[i] = randomVector(numDims);
      }

      return vectors;
    }

    private static float[] randomVector(int numDims) {
      float[] vector = new float[numDims];
      for(int i = 0; i < numDims; i++) {
        vector[i] = random().nextFloat();
      }

      return vector;
    }

    public static QueryResult assertRecall(Directory dir, int expectSize, int topK, float[] value, boolean forceEqual,
                                       VectorValues.VectorIndexType type) throws IOException {
      try (IndexReader reader = DirectoryReader.open(dir)) {
        final ExecutorService es = Executors.newCachedThreadPool(new NamedThreadFactory("HNSW&IVFFLAT"));
        IndexSearcher searcher = new IndexSearcher(reader, es);
        Query query = type == VectorValues.VectorIndexType.HNSW ? new KnnGraphQuery(KNN_VECTOR_FIELD, value, topK) :
            new KnnIvfFlatQuery(KNN_VECTOR_FIELD, value, topK);

        long startTime = System.currentTimeMillis();
        TopDocs result = searcher.search(query, expectSize);
        long costTime = System.currentTimeMillis() - startTime;

        int totalRecallCnt = 0, exactRecallCnt = 0;
        for (LeafReaderContext ctx : reader.leaves()) {
          VectorValues vector = ctx.reader().getVectorValues(KNN_VECTOR_FIELD);
          for (ScoreDoc doc : result.scoreDocs) {
            if (vector.seek(doc.doc - ctx.docBase)) {
              ++totalRecallCnt;
              if (forceEqual) {
                assertEquals(0, Arrays.compare(value, vector.vectorValue()));
                ++exactRecallCnt;
              } else {
                if (Arrays.equals(value, vector.vectorValue())) {
                  ++exactRecallCnt;
                }
              }
            }
          }
        }
        assertEquals(expectSize, totalRecallCnt);

        es.shutdown();

        return new QueryResult(exactRecallCnt, costTime);
      }
    }
  }

  public static final class QueryResult {
    public int recallCnt;
    public long costTime;

    public QueryResult(int recall, long costTime) {
      this.recallCnt = recall;
      this.costTime = costTime;
    }
  }
}
