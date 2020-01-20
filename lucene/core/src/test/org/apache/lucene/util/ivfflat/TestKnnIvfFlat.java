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
import java.util.Random;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.LuceneTestCase;

/** Tests indexing of a ivf-flat */
public class TestKnnIvfFlat extends LuceneTestCase {

  private static final String KNN_IVF_FLAT_FIELD = "vector";

  private Random random = new Random();

  /**
   * Basic test of creating documents in a graph
   */
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

  private void assertConsistentIvfFlat(IndexWriter iw, float[][] values) throws IOException {
    int totalIvfFlatDocs = 0;
    try (DirectoryReader dr = DirectoryReader.open(iw)) {
      for (LeafReaderContext ctx: dr.leaves()) {
        LeafReader reader = ctx.reader();
        VectorValues vectorValues = reader.getVectorValues(KNN_IVF_FLAT_FIELD);
        IvfFlatValues ivfFlatValues = reader.getIvfFlatValues(KNN_IVF_FLAT_FIELD);
        assertTrue((vectorValues == null) == (ivfFlatValues == null));
        if (vectorValues == null) {
          continue;
        }

        boolean hasVector = false;
        for (int i = 0; i < reader.maxDoc(); i++) {
          int id = Integer.parseInt(reader.document(i).get("id"));
          if (values[id] == null) {
            ++totalIvfFlatDocs;
            // documents without IvfFlatValues have no vectors or neighbors
            assertNotEquals("document " + id + " was not expected to have values", i, vectorValues.advance(i));
            assertNotEquals(i, ivfFlatValues.advance(i));
          } else {
            hasVector = true;
            // documents with KnnGraphValues have the expected vectors
            int doc = vectorValues.advance(i);
            assertEquals("doc " + i + " with id=" + id + " has no vector value", i, doc);
            float[] scratch = vectorValues.vectorValue();
            assertArrayEquals("vector did not match for doc " + i + ", id=" + id + ": " + Arrays.toString(scratch),
                values[id], scratch, 0f);
          }
        }

        if (hasVector) {
          int[] centroids = ivfFlatValues.getCentroids();
          for (int docId : centroids) {
            ivfFlatValues.advance(docId);
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

  private List<float[]> randomVectors(int numDocs, int numDims) {
    List<float[]> vectors = new ArrayList<>(numDocs);
    for (int i = 0; i < numDocs; ++i) {
      vectors.add(randomVector(numDims));
    }

    return vectors;
  }

  private float[] randomVector(int numDims) {
    float[] vector = new float[numDims];
    for(int i = 0; i < numDims; i++) {
      vector[i] = random.nextFloat();
    }

    return vector;
  }
}
