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

import java.io.IOException;

import org.apache.lucene.index.IvfFlatValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorValues;

/**
 * Encodes/decodes per-document vector and indexed ivf-flat for approximate nearest neighbor search.
 */
public abstract class IvfFlatIndexFormat {

  /** Sole constructor */
  protected IvfFlatIndexFormat() {
  }

  /**
   * Returns a {@link IvfFlatIndexWriter} to write the vectors and ivfflat to the index.
   */
  public abstract IvfFlatIndexWriter fieldsWriter(SegmentWriteState state) throws IOException;

  /**
   * Returns a {@link IvfFlatIndexReader} to read the vectors and ivfflat from the index.
   */
  public abstract IvfFlatIndexReader fieldsReader(SegmentReadState state) throws IOException;

  public static final IvfFlatIndexFormat EMPTY = new IvfFlatIndexFormat() {

    /**
     * Returns a {@link KnnGraphWriter} to write the vectors and knn-graph to the index.
     */
    @Override
    public IvfFlatIndexWriter fieldsWriter(SegmentWriteState state) {
      throw new UnsupportedOperationException("Attempt to write EMPTY IvfFlat values: maybe you forgot to use codec=Lucene90");
    }

    /**
     * Returns a {@link KnnGraphReader} to read the vectors and knn-graph from the index.
     */
    @Override
    public IvfFlatIndexReader fieldsReader(SegmentReadState state) {
      return new IvfFlatIndexReader() {

        /**
         * Return the memory usage of this object in bytes. Negative values are illegal.
         */
        @Override
        public long ramBytesUsed() {
          return 0L;
        }

        /**
         * Closes this stream and releases any system resources associated
         * with it. If the stream is already closed then invoking this
         * method has no effect.
         *
         * <p> As noted in {@link AutoCloseable#close()}, cases where the
         * close may fail require careful attention. It is strongly advised
         * to relinquish the underlying resources and to internally
         * <em>mark</em> the {@code Closeable} as closed, prior to throwing
         * the {@code IOException}.
         *
         */
        @Override
        public void close() {

        }

        /**
         * Checks consistency of this reader.
         * <p>
         * Note that this may be costly in terms of I/O, e.g.
         * may involve computing a checksum value against large data files.
         *
         * @lucene.internal
         */
        @Override
        public void checkIntegrity() {

        }

        /**
         * Returns the {@link VectorValues} for the given {@code field}
         */
        @Override
        public VectorValues getVectorValues(String field) {
          return VectorValues.EMPTY;
        }

        /**
         * Returns the {@link IvfFlatValues} for the given {@code field}
         *
         * @param field
         */
        @Override
        public IvfFlatValues getIvfFlatValues(String field) throws IOException {
          return IvfFlatValues.EMPTY;
        }
      };
    }
  };
}
