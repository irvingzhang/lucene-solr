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

package org.apache.lucene.codecs.lucene90;

import java.io.IOException;

import org.apache.lucene.codecs.IvfFlatIndexFormat;
import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

public class Lucene90IvfFlatIndexFormat extends IvfFlatIndexFormat {
  static final String META_CODEC_NAME = "Lucene90IvfFlatIndexFormatMeta";

  static final String META_EXTENSION = "ifi";

  static final int VERSION_START = 0;

  static final int VERSION_CURRENT = VERSION_START;
  /**
   * Returns a {@link IvfFlatIndexWriter} to write the vectors and ivfflat to the index.
   */
  @Override
  public IvfFlatIndexWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene90IvfFlatIndexWriter(state);
  }

  /**
   * Returns a {@link IvfFlatIndexReader} to read the vectors and ivfflat from the index.
   *
   * @param state
   */
  @Override
  public IvfFlatIndexReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene90IvfFlatIndexReader(state);
  }
}
