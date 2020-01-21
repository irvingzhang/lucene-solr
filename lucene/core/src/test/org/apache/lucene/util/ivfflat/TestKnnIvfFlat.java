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
import java.util.Arrays;
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
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnIvfFlatQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.LuceneTestCase;
import org.apache.lucene.util.NamedThreadFactory;

/** Tests indexing of a ivf-flat */
public class TestKnnIvfFlat extends LuceneTestCase {

  private static final String KNN_IVF_FLAT_FIELD = "vector";

  public void testBasic() throws Exception {
    try (Directory dir = newDirectory(); IndexWriter iw = new IndexWriter(
        dir, newIndexWriterConfig(null).setCodec(Codec.forName("Lucene90")))) {
      int numDoc = atLeast(20);
      int dimension = atLeast(10);
      float[][] values = new float[numDoc][];
      for (int i = 0; i < numDoc; i++) {
        if (random().nextBoolean()) {
          values[i] = new float[dimension];
          for (int j = 0; j < dimension; j++) {
            values[i][j] = random().nextFloat();
          }
        }
        add(iw, i, values[i]);
      }

      assertConsistentIvfFlat(iw, values);
    }
  }

  public void testSingleDocumentRecall() throws Exception {
    try (Directory dir = newDirectory(); IndexWriter iw = new IndexWriter(dir,
        newIndexWriterConfig(null).setCodec(Codec.forName("Lucene90")))) {
      float[][] values = new float[][]{new float[]{0, 1, 2}};
      add(iw, 0, values[0]);
      assertConsistentIvfFlat(iw, values);
      iw.commit();
      assertConsistentIvfFlat(iw, values);

      assertRecall(dir, 1, 1, values[0], true);
    }
  }

  public void testSearch() throws Exception {
    int numDocs = 50000;
    int dimension = 100;
    float[][] randomVectors = randomVectors(numDocs, dimension);

    try (Directory dir = newDirectory(); IndexWriter iw = new IndexWriter(
        dir, newIndexWriterConfig(null).setMaxBufferedDocs(1000000).setCodec(Codec.forName("Lucene90")))) {
      for (int i = 0; i < numDocs; ++i) {
        add(iw, i, randomVectors[i]);
      }

      iw.commit();
      assertConsistentIvfFlat(iw, randomVectors);

      long totalCostTime = 0;
      int totalRecallCnt = 0;
      QueryResult result;
      int testRecall = Math.min(2000, numDocs);
      for (int i = 0; i < testRecall; ++i) {
        result = assertRecall(dir, 1, 1, randomVectors[i], false);
        totalCostTime += result.costTime;
        totalRecallCnt += result.recallCnt;
      }

      System.out.println("Total number of docs -> " + numDocs + ", dimension -> " + dimension +
          ", recall experiments -> " + testRecall + ", exact recall times -> " + totalRecallCnt +
          ", total search time -> " + totalCostTime + "msec, avg search time -> " +
          1.0F * totalCostTime / testRecall + "msec, recall percent -> " +
          100.0F * totalRecallCnt / testRecall + "%");
    }
  }

  private void assertConsistentIvfFlat(IndexWriter iw, float[][] values) throws IOException {
    int totalIvfFlatDocs = 0;
    try (DirectoryReader dr = DirectoryReader.open(iw)) {
      for (LeafReaderContext ctx: dr.leaves()) {
        LeafReader reader = ctx.reader();
        VectorValues vectorValues = reader.getVectorValues(KNN_IVF_FLAT_FIELD);
        IvfFlatValues ivfFlatValues = reader.getIvfFlatValues(KNN_IVF_FLAT_FIELD);
        assertEquals((vectorValues == null), (ivfFlatValues == null));
        if (vectorValues == null) {
          continue;
        }

        boolean hasVector = false;
        for (int i = 0; i < reader.maxDoc(); i++) {
          int id = Integer.parseInt(Objects.requireNonNull(reader.document(i).get("id")));
          if (values[id] == null) {
            ++totalIvfFlatDocs;
            // documents without IvfFlatValues have no vectors or neighbors
            assertNotEquals("document " + id + " was not expected to have values", i, vectorValues.advance(i));
          } else {
            hasVector = true;
          }
        }

        if (hasVector) {
          int[] centroids = ivfFlatValues.getCentroids();
          assertNotNull(centroids);

          for (int docId : centroids) {
            IntsRef ivfLink = ivfFlatValues.getIvfLink(docId);
            totalIvfFlatDocs += ivfLink.length;
          }
        }
      }
    }

    assertEquals(values.length, totalIvfFlatDocs);
  }

  private void add(IndexWriter iw, int id, float[] vector) throws IOException {
    Document doc = new Document();
    if (vector != null) {
      doc.add(new VectorField(KNN_IVF_FLAT_FIELD, vector, VectorValues.DistanceFunction.EUCLIDEAN,
          VectorValues.VectorIndexType.IVFFLAT));
    }
    doc.add(new StringField("id", Integer.toString(id), Field.Store.YES));
    iw.addDocument(doc);
  }

  private float[][] randomVectors(int numDocs, int numDims) {
    float[][] vectors = new float[numDocs][];
    for (int i = 0; i < numDocs; ++i) {
      vectors[i] = randomVector(numDims);
    }

    return vectors;
  }

  private float[] randomVector(int numDims) {
    float[] vector = new float[numDims];
    for(int i = 0; i < numDims; i++) {
      vector[i] = random().nextFloat();
    }

    return vector;
  }

  private QueryResult assertRecall(Directory dir, int expectSize, int topK, float[] value, boolean forceEqual) throws IOException {
    try (IndexReader reader = DirectoryReader.open(dir)) {
      final ExecutorService es = Executors.newCachedThreadPool(new NamedThreadFactory("HNSW"));
      IndexSearcher searcher = new IndexSearcher(reader, es);
      Query query = new KnnIvfFlatQuery(KNN_IVF_FLAT_FIELD, value, topK);

      long startTime = System.currentTimeMillis();
      TopDocs result = searcher.search(query, expectSize);
      long costTime = System.currentTimeMillis() - startTime;

      assertEquals(expectSize, result.scoreDocs.length);

      int totalRecallCnt = 0, exactRecallCnt = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        VectorValues vector = ctx.reader().getVectorValues(KNN_IVF_FLAT_FIELD);
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

  private static final class QueryResult {
    int recallCnt;
    long costTime;

    QueryResult(int recall, long costTime) {
      this.recallCnt = recall;
      this.costTime = costTime;
    }
  }
}
