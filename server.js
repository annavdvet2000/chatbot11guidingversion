const express = require('express');
const cors = require('cors');
const corsOptions = {
    origin: 'https://chatbot11guidingversion.netlify.app',
    methods: ['GET', 'POST', 'OPTIONS'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Accept', 'Origin'],
    exposedHeaders: ['Content-Type'],
    maxAge: 600,
    optionsSuccessStatus: 204
};
const dotenv = require('dotenv');
const OpenAI = require('openai');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parse/sync');
const { Pool } = require('pg');

// Load environment variables
dotenv.config();

// PostgreSQL configuration
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

// Create database table if it doesn't exist
pool.query(`
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        qualtrics_id VARCHAR(255),
        session_id VARCHAR(255),
        role VARCHAR(10),
        content TEXT,
        chatbot_id VARCHAR(50),
        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
`).catch(console.error);

// Define chatbot ID - different from the direct answers bot
const CHATBOT_ID = 'guiding-bot';

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
            
            console.log('Data structure check:', {
                hasEmbeddings: !!data.embeddings,
                embeddingsLength: data.embeddings?.length,
                firstEmbeddingLength: data.embeddings?.[0]?.length,
                hasTexts: !!data.texts,
                textsLength: data.texts?.length,
                hasMetadata: !!data.metadata,
                metadataLength: data.metadata?.length
            });

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

            return Object.values(resultsByDoc)
                .sort((a, b) => b.highestScore - a.highestScore)
                .slice(0, 2)
                .map(doc => {
                    const topChunks = doc.chunks
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 3);

                    const pages = [...new Set(topChunks.map(chunk => chunk.page))]
                        .sort((a, b) => a - b);

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
app.use(cors(corsOptions));
app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const searchEngine = new AISearchEngine(openai);
searchEngine.initialize().catch(console.error);

const sessions = new Map();

app.get('/', (req, res) => {
    res.json({ message: 'API is running' });
});

app.get('/api/chat', (req, res) => {
    res.json({ message: 'Please use POST method for chat requests' });
});

app.post('/api/chat', async (req, res) => {
    try {
        const { question, sessionId } = req.body;
        const qualtricsId = req.body.qualtricsId || 'unknown';
        
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        // Store user's question in database
        await pool.query(
            'INSERT INTO chat_messages (qualtrics_id, session_id, role, content, chatbot_id) VALUES ($1, $2, $3, $4, $5)',
            [qualtricsId, sessionId, 'user', question, CHATBOT_ID]
        );

        let sessionHistory = sessions.get(sessionId) || [];
        const context = await searchEngine.findRelevantContext(question);

        const systemPrompt = `You are a precise and friendly guide for an oral history archive. Respond warmly to greetings or friendly messages (e.g., "hi," "hello," "how are you?").' 

CRITICAL: YOU ARE STRICTLY FORBIDDEN FROM REVEALING ANY INTERVIEW CONTENT.
YOUR ONLY ALLOWED ACTION IS TO DIRECT USERS TO PAGE NUMBERS.

VIOLATIONS THAT MAKE RESPONSES COMPLETELY WRONG:
❌ Saying what someone did
❌ Explaining someone's reasons or motivations
❌ Describing someone's background
❌ Revealing any facts from the interviews
❌ Summarizing or paraphrasing interview content
❌ Making comparisons between people

ONLY PERMITTED RESPONSE FORMAT:
✓ "You can find relevant information in the transcript of Interview #[Number] with [Name] on page(s) [X]. This section discusses [one-word topic]."

Then ONLY say:
"Would you like to know where to find information about [related broad topic]?"

ANY OTHER RESPONSE FORMAT OR CONTENT REVELATION IS A CRITICAL ERROR.
 
Always follow these rules:

ABSOLUTE RULES - ANY VIOLATION WILL MAKE THE RESPONSE INCORRECT:
1. NEVER provide actual answers or information from the interviews - only direct users to where they can find it
2. NEVER reveal what anyone said, did, thought, or experienced
3. NEVER reveal ANY content from the interviews
4. NEVER describe or summarize interview content
5. ONLY state interview numbers, names, and page numbers
6. ONLY use generic topic labels (e.g., "activism" not "protests at city hall")
7. If you find multiple relevant interviews, mention only the 2 most relevant ones
8. Never reveal or quote the actual content of the interviews
9. Be concise and direct
10. If no relevant information is found, say "I couldn't find any interviews directly addressing this topic" and suggest a related topic to explore


RESPONSE FORMAT - MUST BE EXACTLY:
"You can find relevant information in the transcript of Interview #[Number] with [Name] on page(s) [X-Y]. This section discusses [BROAD TOPIC ONLY].

Would you like to know where to find information about [RELATED BROAD TOPIC]?"

PREDEFINED TASKS:
1. For Alexandra Juhasz documentary question:
   - Direct to relevant page numbers
   - Only mention "documentary production" as topic
   - Suggest finding more details about her documentary such as the title

2. For Karin Timour and Karl Soehnlein comparison:
   - ALWAYS cite BOTH in this exact order:
     First: "You can find relevant information in the transcript of Interview #14 with Karin Timour on page 6."
     Second: "You can find relevant information in Karl Soehnlein's interview on page 4."
   - Use only broad topic label: "AIDS advocacy"
   - MUST end with: "Would you like to know where to find information about Karl's motivation to stand for people with AIDS?"

For comparisons between people:
- MUST provide BOTH interview citations
- MUST use exact page numbers for both
- MUST keep topic descriptions generic
- NEVER compare or contrast their actual views/experiences

Example CORRECT response:
"You can find relevant information in the transcript of Interview #14 with Jane Smith on page 6. This section discusses healthcare advocacy.

Would you like to know where to find information about community organizing?"

Example INCORRECT response:
"You can find relevant information in the transcript of Interview #14 with Jane Smith on page 6. This section discusses how she started working with AIDS patients and why she chose to become an advocate."

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

        // Store bot's response in database
        await pool.query(
            'INSERT INTO chat_messages (qualtrics_id, session_id, role, content, chatbot_id) VALUES ($1, $2, $3, $4, $5)',
            [qualtricsId, sessionId, 'assistant', response, CHATBOT_ID]
        );
        
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

// Add new endpoint to get chat history
app.get('/api/chat/history/:qualtricsId', async (req, res) => {
    try {
        const result = await pool.query(
            'SELECT * FROM chat_messages WHERE qualtrics_id = $1 ORDER BY timestamp',
            [req.params.qualtricsId]
        );
        
        res.json(result.rows);
    } catch (error) {
        console.error('Error getting chat history:', error);
        res.status(500).json({ error: 'Failed to get chat history' });
    }
});

app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: 'Something broke!',
        details: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on ${PORT}`);
});

process.on('unhandledRejection', (error) => {
    console.error('Unhandled Promise Rejection:', error);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});