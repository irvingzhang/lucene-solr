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

package org.apache.lucene.benchmark.vector;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnIvfFlatQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import py4j.GatewayServer;

public class PythonEntryPoint {
  private static final String INDEX_NAME = "/tmp/ivfflat";

  private static final String ID_FIELD = "id";

  private static final String VECTOR_FIELD = "vector";

  private VectorValues.DistanceFunction distanceFunction;

  private Directory directory;

  private IndexWriter indexWriter;

  private IndexReader indexReader;

  public static void main(String[] args) {
    final GatewayServer gatewayServer = new GatewayServer(new PythonEntryPoint());
    gatewayServer.start();
    System.out.println("Gateway Server Started");
  }

  public void prepareIndex(String function) throws IOException {
    this.distanceFunction = VectorValues.DistanceFunction.valueOf(function);

    safeDelete(INDEX_NAME);

    this.directory = NIOFSDirectory.open(Paths.get(INDEX_NAME));

    this.indexWriter = new IndexWriter(directory, new IndexWriterConfig().setOpenMode(
        IndexWriterConfig.OpenMode.CREATE).setCodec(Codec.forName("Lucene90")));
  }

  public void indexBatch(int startId, final List<List<Number>> vectors) throws IOException {
    int id = startId;
    for (List<Number> vector : vectors) {
      final Document doc = new Document();
      doc.add(new StoredField(ID_FIELD, id++));

      final float[] point = convertToArray(vector);
      doc.add(new VectorField(VECTOR_FIELD, point, distanceFunction));
      indexWriter.addDocument(doc);
    }

    indexWriter.flush();
  }

  public void forceMerge() throws IOException {
    indexWriter.forceMerge(1);
    indexWriter.close();
  }

  public void openReader() throws IOException {
    indexReader = DirectoryReader.open(directory);
  }

  public List<Integer> search(final List<Number> query, int k, int nProbe) throws IOException {
    final IndexSearcher searcher = new IndexSearcher(indexReader);

    final float[] value = convertToArray(query);
    final KnnIvfFlatQuery graphQuery = new KnnIvfFlatQuery(VECTOR_FIELD, value, nProbe);

    final TopDocs topDocs = searcher.search(graphQuery, k);

    final List<Integer> result = new ArrayList<>(k);
    for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
      final Document doc = indexReader.document(scoreDoc.doc);
      final IndexableField field = doc.getField(ID_FIELD);

      assert field != null;
      result.add(field.numericValue().intValue());
    }

    return result;
  }

  private float[] convertToArray(final List<Number> vector) {
    final float[] point = new float[vector.size()];
    IntStream.range(0, vector.size()).forEach(i -> point[i] = vector.get(i).floatValue());
    return point;
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
}
