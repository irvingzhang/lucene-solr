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

package org.apache.lucene.codecs;

import java.io.Closeable;
import java.io.IOException;

import org.apache.lucene.index.KnnGraphValues;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.Accountable;

/**
 * Reads vectors and knn-graph from an index.
 */
public abstract class KnnGraphReader implements Closeable, Accountable {

  /** Sole constructor */
  protected KnnGraphReader() {}

  /**
   * Checks consistency of this reader.
   * <p>
   * Note that this may be costly in terms of I/O, e.g.
   * may involve computing a checksum value against large data files.
   * @lucene.internal
   */
  public abstract void checkIntegrity() throws IOException;

  /** Returns the {@link VectorValues} for the given {@code field} */
  public abstract VectorValues getVectorValues(String field) throws IOException;

  /** Returns the {@link KnnGraphValues} for the given {@code field} */
  public abstract KnnGraphValues getGraphValues(String field) throws IOException;

  /**
   * Returns an instance optimized for merging. This instance may only be
   * consumed in the thread that called {@link #getMergeInstance()}.
   * <p>
   * The default implementation returns {@code this} */
  public KnnGraphReader getMergeInstance() {
    return this;
  }
}
