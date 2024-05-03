def get_text_chunks(text, chunk_size=10000):    #chunk_size=This parameter specifies the maximum size of each chunk
  chunks = []
  start_index = 0
  while start_index < len(text):
    end_index = min(start_index + chunk_size, len(text))
    chunk = text[start_index:end_index]
    chunks.append(chunk)
    start_index = end_index
  return chunks