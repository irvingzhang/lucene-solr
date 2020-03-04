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

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
import org.apache.lucene.util.LuceneTestCase;

public class KnnIvfPerformTester extends LuceneTestCase {
  private static final String IVFFLAT_INDEX_DIR = "/tmp/ivfPerf";

  /**
   * args[0] should be the base dataset path, args[1] should be the test query file, args[2] should be the ground-truth file path.
   */
  public static void main(String[] args) {
    if (args.length < 3) {
      PrintHelp();
    }

    try {
      final List<float[]> siftDataset = SiftDataReader.fvecReadAll(args[0]);
      assertNotNull(siftDataset);

      final List<float[]> queryDataset = SiftDataReader.fvecReadAll(args[1]);
      assertNotNull(queryDataset);

      final List<int[]> groundTruthVects = SiftDataReader.ivecReadAll(args[2], queryDataset.size());
      assertNotNull(groundTruthVects);

      boolean success = false;
      for (int i = 0; !success && i < 10; ++i) { /// max retry 10 times
        success = runCase(siftDataset.size(), siftDataset.get(0).length,
            siftDataset, VectorValues.VectorIndexType.IVFFLAT, IVFFLAT_INDEX_DIR, queryDataset, groundTruthVects,
            new int[]{5, 10, 20, 50, 100});
      }
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  public static boolean runCase(int numDoc, int dimension, List<float[]> randomVectors,
                                VectorValues.VectorIndexType type, String indexDir, final List<float[]> queryDataset,
                                final List<int[]> groundTruthVects, int[] centroids) {
    safeDelete(indexDir);

    try (Directory dir = FSDirectory.open(Paths.get(indexDir)); IndexWriter iw = new IndexWriter(
        dir, new IndexWriterConfig(new StandardAnalyzer()).setSimilarity(new AssertingSimilarity(new RandomSimilarity(new Random())))
        .setMaxBufferedDocs(1500000).setRAMBufferSizeMB(4096).setMergeScheduler(new SerialMergeScheduler()).setUseCompoundFile(false)
        .setReaderPooling(false).setCodec(Codec.forName("Lucene90")))) {
      long addStartTime = System.currentTimeMillis();
      for (int i = 0; i < numDoc; ++i) {
        TestKnnIvfFlat.KnnTestHelper.add(iw, i, randomVectors.get(i), type);
      }

      long addEndTime = System.currentTimeMillis();
      iw.commit();
      long commitEndTime = System.currentTimeMillis();

      iw.forceMerge(1);
      long forceEndTime = System.currentTimeMillis();

      System.out.println("[***" + type + "***] [ADD] cost " + (addEndTime - addStartTime) + " msec, [COMMIT] cost "
          + (commitEndTime - addEndTime) + " msec, [ForceMerge(1)] cost " + (forceEndTime - commitEndTime) + " msec, total cost "
          + (forceEndTime - commitEndTime) + " msec");

      TestKnnIvfFlat.KnnTestHelper.assertConsistent(iw, randomVectors, type);

      if (centroids != null && centroids.length > 0) {
        for (int centroid : centroids) {
          TestKnnIvfFlat.assertResult(numDoc, dimension, type, centroid, dir, queryDataset, groundTruthVects,
              randomVectors, groundTruthVects.get(0).length);
        }
      } else {
        TestKnnIvfFlat.assertResult(numDoc, dimension, type, 50, dir, queryDataset, groundTruthVects,
            randomVectors, groundTruthVects.get(0).length);
      }
    } catch (IOException e) {
      e.printStackTrace();
      return false;
    }

    safeDelete(indexDir);

    return true;
  }

  private static void safeDelete(final String indexDir) {
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
    System.err.println("Usage: KnnIvfPerformTester ${baseVecPath} ${queryVecPath} ${groundTruthPath}");
    System.exit(1);
  }

  public static final class SiftDataReader {
    private SiftDataReader() {
    }

    public static List<float[]> fvecReadAll(final String fileName) throws IOException {
      final List<float[]> vectors = new ArrayList<>();
      try (FileInputStream stream = new FileInputStream(fileName);
           InputStream fileStream = new DataInputStream(stream)) {
        byte[] allBytes = fileStream.readAllBytes();
        ByteBuffer byteBuffer = ByteBuffer.wrap(allBytes);
        while (byteBuffer.hasRemaining()) {
          int vecDims = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
          assert vecDims > 0;

          float[] vec = new float[vecDims];
          for (int i = 0; i < vecDims; ++i) {
            vec[i] = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getFloat();
          }
          vectors.add(vec);
        }
      }

      return vectors;
    }

    public static List<int[]> ivecReadAll(final String fileName, int expectSize) throws IOException {
      final List<int[]> vectors = new ArrayList<>(expectSize);
      try (FileInputStream stream = new FileInputStream(fileName);
           InputStream fileStream = new DataInputStream(stream)) {
        byte[] allBytes = fileStream.readAllBytes();
        ByteBuffer byteBuffer = ByteBuffer.wrap(allBytes);
        while (byteBuffer.hasRemaining()) {
          int vecDims = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
          assert vecDims > 0;

          int[] vec = new int[vecDims];
          for (int i = 0; i < vecDims; ++i) {
            vec[i] = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
          }
          vectors.add(vec);
        }
      }

      return vectors;
    }
  }
}
