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

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.IvfFlatIndexFormat;
import org.apache.lucene.codecs.IvfFlatIndexReader;
import org.apache.lucene.codecs.IvfFlatIndexWriter;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.DataOutput;

/**
 * Lucene 9.0 IVFFlat index format.
 * <p>The centroid and the associated points (clustered by K-Means clustering) are write into the file with suffix <tt>.ifi</tt>.</p>
 * <p>
 *   IVFFlat file (.ifi) --&gt; Header,FieldNumber,VectorDataOffset,VectorDataLength,CentroidSize, &lt;Centroid,
 *   IVFListSize,FirstDocID,FirstDocOrder, &lt;DocIDDelta,DocOrder&gt; <sup>IVFListSize-1</sup>&gt;<sup>CentroidSize</sup>,Footer
 * </p>
 * <p>Field types:</p>
 * <ul>
 *   <li>Header --&gt; {@link CodecUtil#checkIndexHeader IndexHeader}</li>
 *   <li>FieldNumber,CentroidSize,IVFListSize --&gt; {@link DataOutput#writeInt Int}</li>
 *   <li>VectorDataOffset,VectorDataLength --&gt; {@link DataOutput#writeVLong VLong}</li>
 *   <li>Centroid,FirstDocID,FirstDocOrder,DocIDDelta,DocOrder --&gt; {@link DataOutput#writeVInt VInt}</li>
 *   <li>Footer --&gt; {@link CodecUtil#writeFooter CodecFooter}</li>
 * </ul>
 * Field Descriptions:
 * <ul>
 *   <li>FieldNumber: the field's number. Note that unlike previous versions of
 *       Lucene, the fields are not numbered implicitly by their order in the
 *       file, instead explicitly.</li>
 *   <li>Centroid: the center point of k-means.</li>
 * </ul>
 *
 * @lucene.experimental
 */
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
