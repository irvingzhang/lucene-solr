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

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.similarities.AssertingSimilarity;
import org.apache.lucene.search.similarities.RandomSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class KnnIvfAndGraphPerformTester extends LuceneTestCase {
  private static final String IVFFLAT_INDEX_DIR = "/tmp/ivfflat";

  private static final String HNSW_INDEX_DIR = "/tmp/hnsw";

  /**
   * args[0] is the specified dataset  path.
   */
  public static void main(String[] args) {
    if (args.length == 0) {
      PrintHelp();
    }

    try {
      final List<float[]> siftDataset = SiftDataReader.readRange(args[0], 0, 2000);
      assertNotNull(siftDataset);

      runCase(siftDataset.size(), siftDataset.get(0).length,
          siftDataset, VectorValues.VectorIndexType.IVFFLAT, IVFFLAT_INDEX_DIR);

      runCase(siftDataset.size(), siftDataset.get(0).length,
          siftDataset, VectorValues.VectorIndexType.HNSW, HNSW_INDEX_DIR);
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  public static void runCase(int numDoc, int dimension, List<float[]> randomVectors,
                       VectorValues.VectorIndexType type, String indexDir) throws Exception {
    safeClose(indexDir);

    try (Directory dir = FSDirectory.open(Paths.get(indexDir)); IndexWriter iw = new IndexWriter(
        dir, new IndexWriterConfig(new StandardAnalyzer()).setSimilarity(new AssertingSimilarity(new RandomSimilarity(new Random())))
        .setMaxBufferedDocs(1000000).setMergeScheduler(new SerialMergeScheduler()).setUseCompoundFile(false)
        .setReaderPooling(false).setCodec(Codec.forName("Lucene90")))) {
      for (int i = 0; i < numDoc; ++i) {
        TestKnnGraphAndIvfFlat.KnnTestHelper.add(iw, i, randomVectors.get(i), type);
      }

      iw.commit();
      TestKnnGraphAndIvfFlat.KnnTestHelper.assertConsistent(iw, randomVectors, type);

      long totalCostTime = 0;
      int totalRecallCnt = 0;
      TestKnnGraphAndIvfFlat.QueryResult result;
      int testRecall = Math.min(numDoc, 2000);
      for (int i = 0; i < testRecall; ++i) {
        result = TestKnnGraphAndIvfFlat.KnnTestHelper.assertRecall(dir, 1, 1, randomVectors.get(i), false, type);
        totalCostTime += result.costTime;
        totalRecallCnt += result.recallCnt;
      }

      System.out.println("[***" + type.toString() + "***] Total number of docs -> " + numDoc +
          ", dimension -> " + dimension + ", recall experiments -> " + testRecall +
          ", exact recall times -> " + totalRecallCnt + ", total search time -> " +
          totalCostTime + "msec, avg search time -> " + 1.0F * totalCostTime / testRecall +
          "msec, recall percent -> " + 100.0F * totalRecallCnt / testRecall + "%");
    }

    safeClose(indexDir);
  }

  private static void safeClose(final String indexDir) {
    try {
      Files.walk(Paths.get(indexDir)).sorted(Comparator.reverseOrder())
          .map(Path::toFile).forEach(File::deleteOnExit);
    } catch (NoSuchFileException ignored) {
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private static void PrintHelp() {
    /// sift data path should be indicated
    System.err.println("Usage: KnnIvfAndGraphPerformTester ${dataPath}");
    System.exit(1);
  }

  public static final class SiftDataReader {
    private SiftDataReader() {
    }

    public static List<float[]> readAll(final String fileName) throws IOException {
      final List<float[]> vectors = new ArrayList<>();
      try (FileInputStream stream = new FileInputStream(fileName);
           InputStream fileStream = new DataInputStream(stream)) {
        while (fileStream.available() > 0) {
          int vecDims = fromByteArray(fileStream.readNBytes(4));
          assert vecDims > 0;

          float[] vec = new float[vecDims];
          for (int i = 0; i < vecDims; ++i) {
            vec[i] = fromByteArray(fileStream.readNBytes(4));
          }
          vectors.add(vec);
        }
      }

      return vectors;
    }

    public static List<float[]> readRange(final String fileName, int from, int to) throws IOException {
      final List<float[]> vectors = new ArrayList<>();
      int idx = 0;
      try (FileInputStream stream = new FileInputStream(fileName);
           InputStream fileStream = new DataInputStream(stream)) {
        while (fileStream.available() > 0) {
          int vecDims = fromByteArray(fileStream.readNBytes(4));
          assert vecDims > 0;

          float[] vec = new float[vecDims];
          for (int i = 0; i < vecDims; ++i) {
            vec[i] = fromByteArray(fileStream.readNBytes(4));
          }

          if (idx >= from && idx < to) {
            vectors.add(vec);
          }

          if (idx == to) {
            return vectors;
          }

          ++idx;
        }
      }

      return vectors;
    }

    /// sift file is stored in little endian order
    private static int fromByteArray(byte[] bytes) {
      return ((bytes[3] & 0xFF) << 24) |
          ((bytes[2] & 0xFF) << 16) |
          ((bytes[1] & 0xFF) << 8) |
          (bytes[0] & 0xFF);
    }
  }
}
