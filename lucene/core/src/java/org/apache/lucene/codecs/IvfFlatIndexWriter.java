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

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorValues;

public abstract class IvfFlatIndexWriter implements Closeable {
  /** Sole constructor */
  protected IvfFlatIndexWriter() {}

  /** Write all values contained in the provided reader */
  public abstract void writeField(FieldInfo fieldInfo, IvfFlatIndexReader reader) throws IOException;

  /** Called once at the end before close */
  public abstract void finish() throws IOException;

  /** Merges in the fields from the readers in
   *  <code>mergeState</code>. The default implementation
   *  calls {@link #mergeOneField} for each field.
   *  Implementations can override this method
   *  for more sophisticated merging. */
  public void merge(MergeState mergeState) throws IOException {
    for (IvfFlatIndexReader reader : mergeState.ivfFlatIndexReaders) {
      if (reader != null) {
        reader.checkIntegrity();
      }
    }

    mergeState.mergeFieldInfos.forEach(fieldInfo -> {
      if (fieldInfo.hasVectorValues()) {
        mergeOneField(fieldInfo, mergeState);
      }
    });

    finish();
  }

  protected abstract void mergeOneField(FieldInfo fieldInfo, MergeState state);
}
