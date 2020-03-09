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

package org.apache.lucene.util.hnsw;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Hierarchical NSW graph that provides efficient approximate nearest neighbor search for high dimensional vectors.
 * This isn't thread-safe.
 * See <a href="https://arxiv.org/abs/1603.09320">this paper</a> for details.
 */
public final class HNSWGraph implements Accountable {

  private final VectorValues.DistanceFunction distFunc;
  private final List<Layer> layers;

  private boolean frozen = false;
  private long bytesUsed;

  public HNSWGraph(VectorValues.DistanceFunction distFunc) {
    this.distFunc = distFunc;
    this.layers = new ArrayList<>();
  }

  Neighbor greedyUpdateNearest(final float[] query, Neighbor ep, int level, VectorValues vectorValues) throws IOException {
    if (level >= layers.size()) {
      throw new IllegalArgumentException("layer does not exist for the level: " + level);
    }

    final Layer layer = layers.get(level);
    float nearestDist = ep.distance();
    int nearestDocId = ep.docId();
    for (;;) {
      int prevNearestDocId = nearestDocId;
      final Collection<Neighbor> friends = layer.getFriends(ep.docId());
      for (Neighbor neighbor : friends) {
        float qDist = distance(query, neighbor.docId(), vectorValues);
        if (qDist < nearestDist) {
          nearestDist = qDist;
          nearestDocId = neighbor.docId();
        }
      }

      if (nearestDocId == prevNearestDocId) {
        return new ImmutableNeighbor(nearestDocId, nearestDist);
      }
    }
  }

  void addLinksStartingFrom(int docId, final float[] query, final FurthestNeighbors results, int ef,
                            Neighbor ep, int level, VectorValues vectorValues, int mMax) throws IOException {
    results.clear();
    results.add(ep);

    searchLayer(query, results, ef, level, vectorValues);

    final Layer layer = layers.get(level);
    assert layer != null;

    Iterator<Neighbor> iterator;
    if (results.size() > mMax) {
      final TreeSet<Neighbor> neighbors = shrinkNeighbors(layer, results, mMax, vectorValues);
      iterator = neighbors.iterator();
    } else {
      iterator = results.iterator();
    }

    while (iterator.hasNext()) {
      final Neighbor neighbor = iterator.next();
      layer.addLink(docId, neighbor.docId(), mMax, neighbor.distance(), this, vectorValues);
      layer.addLink(neighbor.docId(), docId, mMax, neighbor.distance(), this, vectorValues);
    }
  }

  private TreeSet<Neighbor> shrinkNeighbors(final Layer layer, final FurthestNeighbors results, int mMax,
                                            VectorValues vectorValues) throws IOException {
    assert results.size() > mMax;

    final TreeSet<Neighbor> friends = new TreeSet<>();
    while (results.size() > 0) { /// results are in descending order
      friends.add(results.pop()); /// friends are now in ascending order
    }

    return layer.doShrink(friends, mMax, this, vectorValues);
  }

  /**
   * Searches the nearest neighbors for a specified query at a level.
   *
   * @param query        search query vector
   * @param results      on entry, has enter points to this level. On exit, the nearest neighbors in this level
   * @param ef           the number of nodes to be searched
   * @param level        graph level
   * @param vectorValues vector values
   * @return number of candidates visited
   */
  int searchLayer(float[] query, FurthestNeighbors results, int ef, int level, VectorValues vectorValues) throws IOException {
    if (level >= layers.size()) {
      throw new IllegalArgumentException("layer does not exist for the level: " + level);
    }

    Layer layer = layers.get(level);
    TreeSet<Neighbor> candidates = new TreeSet<>();
    // set of docids that have been visited by search on this layer, used to avoid backtracking
    Set<Integer> visited = new HashSet<>();
    for (Neighbor n : results) {
      candidates.add(n);
      visited.add(n.docId());
    }
    Neighbor f = results.top();
    while (candidates.size() > 0) {
      Neighbor c = candidates.pollFirst();
      assert !c.isDeferred();
      assert !f.isDeferred();
      if (c.distance() > f.distance() && results.size() >= ef) {
        break;
      }
      for (Neighbor e : layer.getFriends(c.docId())) {
        if (visited.contains(e.docId())) {
          continue;
        }
        visited.add(e.docId());
        float dist = distance(query, e.docId(), vectorValues);
        if (dist < f.distance() || results.size() < ef) {
          if (results.size() == ef) {
            results.pop();
          }
          Neighbor n = new ImmutableNeighbor(e.docId(), dist);
          candidates.add(n);
          results.add(n);
          f = results.top();
        }
      }
    }

    //System.out.println("level=" + level + ", visited nodes=" + visited.size());
    //return pickNearestNeighbor(results);
    return visited.size();
  }

  float distance(int doc1, int doc2, VectorValues vectorValues) throws IOException {
    final float[] docValue = getVectorValues(doc1, vectorValues);
    return distance(docValue, doc2, vectorValues);
  }

  private float distance(float[] query, int docId, VectorValues vectorValues) throws IOException {
    final float[] other = getVectorValues(docId, vectorValues);
    return VectorValues.distance(query, other, distFunc);
  }

  private float[] getVectorValues(int docId, VectorValues vectorValues) throws IOException {
    if (!vectorValues.seek(docId)) {
      throw new IllegalStateException("docId=" + docId + " has no vector value");
    }
    return vectorValues.vectorValue();
  }

  public void ensureLevel(int level) {
    if (frozen) {
      throw new IllegalStateException("graph is already freezed!");
    }
    if (level < 0) {
      throw new IllegalArgumentException("level must be a positive integer: " + level);
    }
    for (int l = layers.size(); l <= level; l++) {
      layers.add(new Layer(l));
    }
  }

  public int topLevel() {
    return layers.size() - 1;
  }

  public boolean isEmpty() {
    return layers.isEmpty() || layers.get(0).getNodes().isEmpty();
  }

  public int getFirstEnterPoint() {
    if (layers.isEmpty()) {
      throw new IllegalStateException("the graph has no layers!");
    }
    List<Integer> nodesAtMaxLevel = layers.get(layers.size() - 1).getNodes();
    if (nodesAtMaxLevel.isEmpty()) {
      throw new IllegalStateException("the max level of this graph is empty!");
    }
    return nodesAtMaxLevel.get(0);
  }

  public List<Integer> getEnterPoints() {
    if (layers.isEmpty()) {
      throw new IllegalStateException("the graph has no layers!");
    }
    List<Integer> nodesAtMaxLevel = layers.get(layers.size() - 1).getNodes();
    if (nodesAtMaxLevel.isEmpty()) {
      throw new IllegalStateException("the max level of this graph is empty!");
    }
    return List.copyOf(nodesAtMaxLevel);
  }

  public boolean hasNodes(int level) {
    Layer layer = layers.get(level);
    if (layer == null) {
      throw new IllegalArgumentException("layer does not exist for level: " + level);
    }
    return layer.getNodes().size() > 0;
  }

  public IntsRef getFriends(int level, int node) {
    Layer layer = layers.get(level);
    if (layer == null) {
      throw new IllegalArgumentException("layer does not exist for level: " + level);
    }
    int[] friends = layer.getFriends(node).stream().mapToInt(Neighbor::docId).sorted().toArray();
    return new IntsRef(friends, 0, friends.length);
  }

  public boolean hasFriends(int level, int node) {
    Layer layer = layers.get(level);
    if (layer == null) {
      throw new IllegalArgumentException("layer does not exist for level: " + level);
    }
    return layer.getFriends(node) != Layer.NO_FRIENDS;
  }

  void addNode(int level, int node) {
    if (frozen) {
      throw new IllegalStateException("graph is already freezed!");
    }
    Layer layer = layers.get(level);
    if (layer == null) {
      throw new IllegalArgumentException("layer does not exist for level: " + level);
    }
    layer.addNodeIfAbsent(node);
  }

  /**
   * Connects two nodes; this is supposed to be called when searching
   */
  public void connectNodes(int level, int node1, int node2) {
    if (frozen) {
      throw new IllegalStateException("graph is already freezed!");
    }
    assert level >= 0;
    assert node1 >= 0 && node2 >= 0;
    assert node1 != node2;

    Layer layer = layers.get(level);
    if (layer == null) {
      throw new IllegalArgumentException("layer does not exist for level: " + level);
    }
    layer.connectNodes(node1, node2);
  }

  public void finish() {
    while (!layers.isEmpty() && layers.get(layers.size() - 1).size() == 0) {
      // remove empty top layers
      layers.remove(layers.size() - 1);
    }
    this.frozen = true;
  }

  @Override
  public long ramBytesUsed() {
    if (bytesUsed == 0) {
      bytesUsed = RamUsageEstimator.sizeOfCollection(layers);
    }
    return bytesUsed;
  }

}
