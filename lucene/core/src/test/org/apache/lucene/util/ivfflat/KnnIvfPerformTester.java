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

import java.util.List;

import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.KnnIvfAndGraphPerformTester;
import org.apache.lucene.util.LuceneTestCase;

public class KnnIvfPerformTester extends LuceneTestCase {
  private static final String IVFFLAT_INDEX_DIR = "/tmp/ivfPerf";

  /**
   * args[0] is the specified dataset path.
   */
  public static void main(String[] args) {
    if (args.length == 0) {
      PrintHelp();
    }

    try {
      final List<float[]> siftDataset = KnnIvfAndGraphPerformTester.SiftDataReader.readAll(args[0]);
      assertNotNull(siftDataset);

      KnnIvfAndGraphPerformTester.runCase(siftDataset.size(), siftDataset.get(0).length,
          siftDataset, VectorValues.VectorIndexType.IVFFLAT, IVFFLAT_INDEX_DIR);
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  private static void PrintHelp() {
    /// sift data path should be indicated
    System.err.println("Usage: KnnIvfPerformTester ${dataPath}");
    System.exit(1);
  }
}
