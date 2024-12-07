const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const OpenAI = require('openai');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parse/sync');

// Load environment variables
dotenv.config();

class AISearchEngine {
    constructor(openai) {
        this.openai = openai;
        this.embeddings = [];
        this.texts = [];
        this.metadata = new Map();
        this.chunkMetadata = [];
    }

    async initialize() {
        try {
            const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'embeddings.json'), 'utf8'));
            
            // Debug log to see the structure
            console.log('Data structure check:', {
                hasEmbeddings: !!data.embeddings,
                embeddingsLength: data.embeddings?.length,
                firstEmbeddingLength: data.embeddings?.[0]?.length,
                hasTexts: !!data.texts,
                textsLength: data.texts?.length,
                hasMetadata: !!data.metadata,
                metadataLength: data.metadata?.length
            });

            // Validate data structure before assignment
            if (!data.embeddings || !Array.isArray(data.embeddings)) {
                throw new Error('Invalid embeddings data structure');
            }

            this.embeddings = data.embeddings;
            this.texts = data.texts || [];
            this.chunkMetadata = data.metadata || [];

            const metadataFile = fs.readFileSync(path.join(__dirname, 'metadata.csv'), 'utf8');
            const records = csv.parse(metadataFile, {
                columns: true,
                skip_empty_lines: true
            });
            
            records.forEach((record, index) => {
                const documentId = (index + 1).toString();
                this.metadata.set(documentId, record);
            });
            
            console.log(`Loaded ${this.embeddings.length} embeddings and ${this.metadata.size} metadata records`);
        } catch (error) {
            console.error('Failed to load data:', error);
            throw error;
        }
    }

    async findRelevantContext(question) {
        try {
            const questionEmbedding = await this.getEmbedding(question);
            const similarContent = await this.findSimilarContent(questionEmbedding);
            
            if (!similarContent || similarContent.length === 0) {
                return [];
            }

            // Group similar content by document but keep individual chunks
            const resultsByDoc = {};
            
            similarContent.forEach(item => {
                if (!item.metadata || !item.metadata.source) return;
                
                const docId = item.metadata.source.match(/document(\d+)\.pdf/)?.[1];
                if (!docId) return;

                if (!resultsByDoc[docId]) {
                    const metadata = this.metadata.get(docId);
                    resultsByDoc[docId] = {
                        id: docId,
                        name: metadata?.name || 'Unknown',
                        chunks: [],
                        highestScore: 0
                    };
                }
                
                resultsByDoc[docId].chunks.push({
                    text: item.text,
                    page: item.metadata.page,
                    score: item.score
                });
                
                resultsByDoc[docId].highestScore = Math.max(resultsByDoc[docId].highestScore, item.score);
            });

            // Convert results to final format
            return Object.values(resultsByDoc)
                .sort((a, b) => b.highestScore - a.highestScore) // Sort documents by relevance
                .slice(0, 2) // Limit to top 2 most relevant documents
                .map(doc => {
                    // Sort chunks by relevance score and take top 3 most relevant chunks per document
                    const topChunks = doc.chunks
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 3);

                    // Get unique pages from top chunks
                    const pages = [...new Set(topChunks.map(chunk => chunk.page))]
                        .sort((a, b) => a - b);

                    // Generate page ranges
                    const pageRanges = this.createPageRanges(pages);

                    return {
                        interview: {
                            id: doc.id,
                            name: doc.name,
                            pages: pageRanges,
                            relevanceScore: doc.highestScore,
                            text: topChunks.map(chunk => chunk.text).join('\n\n')
                        }
                    };
                });
        } catch (error) {
            console.error('Error finding relevant context:', error);
            throw error;
        }
    }

    createPageRanges(pages) {
        if (!pages.length) return [];
        
        const ranges = [];
        let rangeStart = pages[0];
        let prev = pages[0];

        for (let i = 1; i <= pages.length; i++) {
            if (i === pages.length || pages[i] !== prev + 1) {
                if (rangeStart === prev) {
                    ranges.push(rangeStart);
                } else {
                    ranges.push(`${rangeStart}-${prev}`);
                }
                if (i < pages.length) {
                    rangeStart = pages[i];
                }
            }
            if (i < pages.length) {
                prev = pages[i];
            }
        }

        return ranges;
    }

    findSimilarContent(queryEmbedding, sourceName = null) {
        if (!this.embeddings || !this.embeddings.length) {
            console.log('No embeddings available');
            return [];
        }

        if (!queryEmbedding || !Array.isArray(queryEmbedding)) {
            console.error('Invalid query embedding');
            return [];
        }

        const similarities = this.embeddings
            .map((emb, idx) => {
                if (!emb || !Array.isArray(emb) || emb.length !== queryEmbedding.length) {
                    console.error(`Invalid embedding at index ${idx}`);
                    return null;
                }

                try {
                    return {
                        score: this.cosineSimilarity(queryEmbedding, emb),
                        text: this.texts[idx] || '',
                        metadata: this.chunkMetadata[idx] || {}
                    };
                } catch (error) {
                    console.error(`Error processing embedding at index ${idx}:`, error);
                    return null;
                }
            })
            .filter(item => item !== null);

        let results = similarities;
        if (sourceName) {
            results = similarities.filter(item => 
                item.metadata && item.metadata.source === sourceName
            );
        }

        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
    }

    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (normA * normB);
    }

    async getEmbedding(text) {
        const response = await this.openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
        });
        return response.data[0].embedding;
    }
}

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const searchEngine = new AISearchEngine(openai);
searchEngine.initialize().catch(console.error);

const sessions = new Map();

app.post('/api/chat', async (req, res) => {
    try {
        const { question, sessionId } = req.body;
        
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        let sessionHistory = sessions.get(sessionId) || [];
        const context = await searchEngine.findRelevantContext(question);

        const systemPrompt = `You are a precise and friendly guide for an oral history archive. Respond warmly to greetings or friendly messages (e.g., "hi," "hello," "how are you?"). 
For more specific questions or topics, guide the user to relevant pages in the archive. If you cannot find relevant context, politely suggest the user provide a more specific query. Always follow these rules:

1. Format your response in exactly this way:
   - First line: For a single page: "You can find relevant information in Interview #[Number] with [Name] on page [X]"
                For multiple pages: "You can find relevant information in Interview #[Number] with [Name] on pages [X-Y]"
   - Second line: A brief explanation of why this section is relevant (1-2 sentences)

2. If you find multiple relevant interviews, mention only the 2 most relevant ones
3. Never reveal or quote the actual content of the interviews
4. Be concise and direct
5. If no relevant information is found, say "I couldn't find any interviews directly addressing this topic" and suggest a related topic to explore

Context: ${JSON.stringify(context)}`;

        const completion = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    role: "system",
                    content: systemPrompt
                },
                ...sessionHistory,
                {
                    role: "user",
                    content: question
                }
            ],
            temperature: 0.7,
            max_tokens: 200
        });

        let response = completion.choices[0].message.content;
        
        sessionHistory = [
            ...sessionHistory,
            { role: "user", content: question },
            { role: "assistant", content: response }
        ].slice(-6);
        
        sessions.set(sessionId, sessionHistory);

        res.json({ response });

    } catch (error) {
        console.error('Error in chat endpoint:', error);
        res.status(500).json({
            error: 'An error occurred while processing your request',
            status: 'error'
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

process.on('unhandledRejection', (error) => {
    console.error('Unhandled Promise Rejection:', error);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});