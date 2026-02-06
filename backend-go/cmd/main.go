package main

import (
	"context"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/samico/lgpd-diff-priv/pb"
)

const (
	uploadPath = "./data"
	workerAddr = "localhost:50051"
)

type PageData struct {
	Response *pb.AnonymizeResponse
	Error    string
}

func main() {
	os.MkdirAll(uploadPath, os.ModePerm)

	mux := http.NewServeMux()

	// Rotas principais
	mux.HandleFunc("GET /", indexHandler)
	mux.HandleFunc("POST /upload", uploadHandler)
	mux.HandleFunc("GET /debug-gui", debugHandler)

	// Servidor de arquivos estáticos
	mux.Handle("GET /data/", http.StripPrefix("/data/", http.FileServer(http.Dir(uploadPath))))

	fmt.Println("==================================================")
	fmt.Println("🚀 BACKEND GO: PRIVACY ENGINE ATIVO")
	fmt.Println("🔗 URL LOCAL: http://localhost:8080")
	fmt.Println("🛠️  DEBUG GUI: http://localhost:8080/debug-gui")
	fmt.Println("==================================================")

	log.Fatal(http.ListenAndServe(":8080", mux))
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("web/index.html")
	if err != nil {
		log.Printf("[ERROR] Erro ao carregar index.html: %v", err)
		http.Error(w, "Template não encontrado", 500)
		return
	}
	tmpl.Execute(w, nil)
}

func debugHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("[DEBUG] Renderizando layout com estilos e dados fictícios...")

	// FORÇA UTF-8 NO CABEÇALHO
	w.Header().Set("Content-Type", "text/html; charset=utf-8")

	mockResp := &pb.AnonymizeResponse{
		OutputPath:      "amostra_final_tse.parquet",
		PrivacyScore:    0.9215,
		UtilityScore:    0.7840,
		EpsilonUsed:     1.0,
		SinglingOutRisk: 0.0120, // FICARÁ VERDE (Seguro)
		LinkabilityRisk: 0.1250, // FICARÁ AMARELO (Moderado)
		InferenceRisk:   0.3640, // FICARÁ VERMELHO (Vulnerável)
		Status:          "ESTADO DE DEBUG // AMBIENTE DE DESENVOLVIMENTO",
		PiiReport: map[string]string{
			"NM_CANDIDATO": "MASKED",
			"NR_CPF":       "HASHED",
			"DS_EMAIL":     "REDACTED",
			"DT_NASC":      "GENERALIZED",
		},
	}

	// Wrapper HTML para o Debug carregar estilos
	fmt.Fprintf(w, `
		<!DOCTYPE html>
		<html lang="pt-br">
		<head>
			<meta charset="UTF-8">
			<script src="https://cdn.tailwindcss.com"></script>
			<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
			<style>
				body { font-family: 'Inter', sans-serif; background-color: #F5F5F5; padding: 50px; }
				.swiss-red { color: #E62B1E; }
			</style>
		</head>
		<body>
			<div class="max-w-4xl mx-auto">
	`)

	tmpl, err := template.ParseFiles("web/result.html")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	tmpl.Execute(w, PageData{Response: mockResp, Error: ""})

	fmt.Fprintf(w, `</div></body></html>`)
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("\n--- [INFO] Nova Requisição de Processamento ---")

	if err := r.ParseMultipartForm(50 << 20); err != nil {
		renderResult(w, nil, "Arquivo muito grande")
		return
	}

	file, header, err := r.FormFile("dataset")
	if err != nil {
		renderResult(w, nil, "Selecione um arquivo válido")
		return
	}
	defer file.Close()

	epsilonStr := r.FormValue("epsilon")
	epsilon, _ := strconv.ParseFloat(epsilonStr, 64)
	if epsilon <= 0 {
		epsilon = 1.0
	}

	filename := filepath.Join(uploadPath, "raw_"+header.Filename)
	out, err := os.Create(filename)
	if err != nil {
		renderResult(w, nil, "Erro ao salvar arquivo")
		return
	}
	defer out.Close()
	io.Copy(out, file)

	absPath, _ := filepath.Abs(filename)

	conn, err := grpc.Dial(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		renderResult(w, nil, "Worker Offline")
		return
	}
	defer conn.Close()

	client := pb.NewPrivacyServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	resp, err := client.ProcessDataset(ctx, &pb.AnonymizeRequest{
		InputPath: absPath,
		Epsilon:   float32(epsilon),
		DetectPii: true,
	})

	if err != nil {
		renderResult(w, nil, "Erro no processamento gRPC")
		return
	}

	renderResult(w, resp, "")
}

func renderResult(w http.ResponseWriter, resp *pb.AnonymizeResponse, errStr string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl, err := template.ParseFiles("web/result.html")
	if err != nil {
		log.Printf("[ERROR] Erro ao carregar result.html: %v", err)
		http.Error(w, "Erro ao renderizar resultado", 500)
		return
	}
	tmpl.Execute(w, PageData{Response: resp, Error: errStr})
}
