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


import java.util.Arrays;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.LuceneTestCase;
import org.apache.lucene.util.TestKnnGraphAndIvfFlat;

/** Tests indexing of a ivf-flat */
public class TestKnnIvfFlat extends LuceneTestCase {

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
        TestKnnGraphAndIvfFlat.KnnTestHelper.add(iw, i, values[i], VectorValues.VectorIndexType.IVFFLAT);
      }

      TestKnnGraphAndIvfFlat.KnnTestHelper.assertConsistent(iw, Arrays.asList(values), VectorValues.VectorIndexType.IVFFLAT);
    }
  }

  public void testSingleDocumentRecall() throws Exception {
    try (Directory dir = newDirectory(); IndexWriter iw = new IndexWriter(dir,
        newIndexWriterConfig(null).setCodec(Codec.forName("Lucene90")))) {
      float[][] values = new float[][]{new float[]{0, 1, 2}};
      TestKnnGraphAndIvfFlat.KnnTestHelper.add(iw, 0, values[0], VectorValues.VectorIndexType.IVFFLAT);
      TestKnnGraphAndIvfFlat.KnnTestHelper.assertConsistent(iw, Arrays.asList(values), VectorValues.VectorIndexType.IVFFLAT);
      iw.commit();
      TestKnnGraphAndIvfFlat.KnnTestHelper.assertConsistent(iw, Arrays.asList(values), VectorValues.VectorIndexType.IVFFLAT);

      try (IndexReader reader = DirectoryReader.open(dir)) {
        TestKnnGraphAndIvfFlat.KnnTestHelper.assertRecall(reader, 1, 1, values[0], true,
            VectorValues.VectorIndexType.IVFFLAT, 50);
      }
    }
  }

  public void testSearch() throws Exception {
    int numDocs = 50000;
    int dimension = 100;
    float[][] randomVectors = TestKnnGraphAndIvfFlat.KnnTestHelper.randomVectors(numDocs, dimension);

    TestKnnGraphAndIvfFlat.runCase(numDocs, dimension, randomVectors, VectorValues.VectorIndexType.IVFFLAT, null);
  }
}
