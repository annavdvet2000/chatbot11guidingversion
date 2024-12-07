const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const { encode } = require('gpt-3-encoder');

// Load environment variables
dotenv.config();

class DocumentProcessor {
    constructor() {
        this.openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
    }

    async processDocuments() {
        try {
            // 1. Read documents from the frontend PDF directory
            const docs = await this.readDocuments();
            console.log(`Found ${docs.length} pages across all documents`);
            
            // 2. Split documents into chunks
            const chunks = await this.splitIntoChunks(docs);
            console.log(`Created ${chunks.length} chunks`);

            // 3. Generate embeddings for each chunk
            const embeddings = await this.generateEmbeddings(chunks);
            
            // 4. Save embeddings and chunks
            await this.saveEmbeddings(embeddings, chunks);
            
            console.log('Embedding generation complete!');
        } catch (error) {
            console.error('Error processing documents:', error);
        }
    }

    async readDocuments() {
        const pdfPath = path.join(__dirname, '..', '..', 'frontend', 'assets', 'pdfs');
        const files = fs.readdirSync(pdfPath).filter(file => file.endsWith('.pdf'));

        const documents = [];

        for (const file of files) {
            const filePath = path.join(pdfPath, file);
            console.log(`Processing ${file}...`);
            
            try {
                const dataBuffer = fs.readFileSync(filePath);
                const pdfData = await pdf(dataBuffer);
                const pages = pdfData.text.split(/\f/); // Split by page delimiter

                pages.forEach((text, pageIndex) => {
                    documents.push({
                        text: text.trim(),
                        title: file,
                        page: pageIndex + 1 // 1-based page numbering
                    });
                });

                console.log(`Processed ${file}: ${pages.length} pages`);
            } catch (error) {
                console.error(`Error processing ${file}:`, error);
            }
        }

        return documents;
    }

    async splitIntoChunks(documents, maxTokens = 500) {
        const chunks = [];

        for (const doc of documents) {
            let currentChunk = '';
            const paragraphs = doc.text.split(/\n\s*\n/); // Split by double newline

            for (const paragraph of paragraphs) {
                const trimmedParagraph = paragraph.trim();
                if (!trimmedParagraph) continue;

                const potentialChunk = currentChunk + '\n' + trimmedParagraph;
                const tokenCount = encode(potentialChunk).length;

                if (tokenCount > maxTokens && currentChunk) {
                    chunks.push({
                        text: currentChunk.trim(),
                        source: doc.title,
                        page: doc.page, // Track page number
                        tokens: encode(currentChunk).length
                    });
                    currentChunk = trimmedParagraph;
                } else {
                    currentChunk = potentialChunk;
                }
            }

            if (currentChunk.trim()) {
                chunks.push({
                    text: currentChunk.trim(),
                    source: doc.title,
                    page: doc.page, // Track page number
                    tokens: encode(currentChunk).length
                });
            }
        }

        return chunks;
    }

    async generateEmbeddings(chunks) {
        const embeddings = [];
        const batchSize = 20;

        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            console.log(`Processing batch ${i / batchSize + 1} of ${Math.ceil(chunks.length / batchSize)}`);

            const batchPromises = batch.map(async chunk => {
                try {
                    const response = await this.openai.embeddings.create({
                        model: "text-embedding-3-small",
                        input: chunk.text,
                    });
                    return {
                        embedding: response.data[0].embedding,
                        metadata: {
                            text: chunk.text,
                            source: chunk.source,
                            page: chunk.page,
                            tokens: chunk.tokens
                        }
                    };
                } catch (error) {
                    console.error(`Error generating embedding for chunk: ${error.message}`);
                    return null;
                }
            });

            const batchResults = await Promise.all(batchPromises);
            embeddings.push(...batchResults.filter(res => res !== null));
            
            if (i + batchSize < chunks.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        return embeddings;
    }

    async saveEmbeddings(embeddings) {
        const outputPath = path.join(__dirname, '..', 'embeddings.json');

        const data = {
            embeddings: embeddings.map(e => e.embedding),
            texts: embeddings.map(e => e.metadata.text), // Added texts array
            metadata: embeddings.map(e => ({
                source: e.metadata.source,
                page: e.metadata.page,
                tokens: e.metadata.tokens
            }))
        };

        fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
        console.log(`Saved embeddings to ${outputPath}`);
    }
}

// Run the embedding generation
const processor = new DocumentProcessor();
processor.processDocuments();