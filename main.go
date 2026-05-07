package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"net/http"
	"os"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// OllamaModel represents a model that communicates with Ollama via its API
type OllamaModel struct {
	baseURL string
	model   string
	client  *http.Client
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents a request to Ollama's chat API
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

// ChatResponse represents a response from Ollama's chat API
type ChatResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Message            Message   `json:"message"`
	Done               bool      `json:"done"`
	TotalDuration      int64     `json:"total_duration"`
	LoadDuration       int64     `json:"load_duration"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int64     `json:"prompt_eval_duration"`
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int64     `json:"eval_duration"`
}

// Name returns the name of the model
func (m *OllamaModel) Name() string {
	return m.model
}

// GenerateContent implements the model.LLM interface
func (m *OllamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	// This is a simplified implementation that doesn't support streaming
	// For now, we'll return a sequence with a single response

	return func(yield func(*model.LLMResponse, error) bool) {
		ollamaMessages := make([]Message, 0, len(req.Contents))

		for _, content := range req.Contents {
			for _, part := range content.Parts {
				if part.Text == "" {
					continue
				}
				ollamaMessages = append(ollamaMessages, Message{
					Role:    content.Role,
					Content: part.Text,
				})
			}
		}

		requestBody := ChatRequest{
			Model:    m.model,
			Messages: ollamaMessages,
			Stream:   true,
		}

		jsonData, err := json.Marshal(requestBody)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal request: %v", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/api/chat", bytes.NewReader(jsonData))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create request: %v", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := m.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to send request to Ollama: %v", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			yield(nil, fmt.Errorf("Ollama returned status code %d", resp.StatusCode))
			return
		}

		var chatResp ChatResponse

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			var chunk ChatResponse
			if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
				continue
			}
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{genai.NewPartFromText(chunk.Message.Content)},
					Role:  genai.RoleModel,
				},
				TurnComplete: chunk.Done,
			}, nil)
			if chunk.Done {
				return
			}
		}

		response := &model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(chatResp.Message.Content)},
				Role:  genai.RoleModel,
			},
			TurnComplete: true,
		}

		yield(response, nil)
	}
}

func main() {
	ctx := context.Background()

	// Check if Ollama is running
	client := &http.Client{
		Timeout: 5 * time.Second,
	}
	_, err := client.Get("http://10.147.17.169:11434")
	if err != nil {
		log.Fatal("Ollama is not running. Please start Ollama first with: ollama run qwen2.5-coder:7b")
	}

	// Create a model that points to Ollama
	ollamaModel := &OllamaModel{
		baseURL: "http://10.147.17.169:11434",
		model:   "qwen2.5-coder:7b",
		client:  &http.Client{Timeout: 60 * time.Second},
	}

	timeAgent, err := llmagent.New(llmagent.Config{
		Name:        "hello_time_agent",
		Model:       ollamaModel,
		Description: "An agent that can interact with the qwen2.5-coder:7b model via Ollama.",
		Instruction: "You are a helpful assistant based on qwen2.5-coder:7b running via Ollama.",
		Tools:       []tool.Tool{}, // Removed GoogleSearch tool since we're not using Gemini
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(timeAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
