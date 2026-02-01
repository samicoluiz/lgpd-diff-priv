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
	mux.HandleFunc("GET /", indexHandler)
	mux.HandleFunc("POST /upload", uploadHandler)
	mux.Handle("GET /data/", http.StripPrefix("/data/", http.FileServer(http.Dir(uploadPath))))

	fmt.Println("[INFO] Backend Go rodando em http://localhost:8080")
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

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("\n--- [INFO] Nova Requisição de Upload ---")

	// 1. Parse do formulário multipart (Limite de 32MB para o upload)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		log.Printf("[ERROR] Erro ao processar multipart: %v", err)
		renderResult(w, nil, "Erro ao processar formulário")
		return
	}

	// 2. Receber o arquivo vindo do HTML
	file, header, err := r.FormFile("dataset")
	if err != nil {
		log.Printf("[ERROR] Arquivo não encontrado no formulário: %v", err)
		renderResult(w, nil, "Por favor, selecione um arquivo válido")
		return
	}
	defer file.Close()

	// 3. Capturar o valor do Epsilon (Slider/Input)
	epsilonStr := r.PostFormValue("epsilon")
	if epsilonStr == "" {
		// Tenta pegar via FormValue caso o multipart tenha se comportado de forma estranha
		epsilonStr = r.FormValue("epsilon")
	}

	fmt.Printf("[DEBUG] Epsilon string recebida: '%s'\n", epsilonStr)

	epsilon, err := strconv.ParseFloat(epsilonStr, 64)
	if err != nil || epsilon <= 0 {
		fmt.Printf("[WARN] Epsilon inválido ou ausente (%s). Usando 1.0 como padrão.\n", epsilonStr)
		epsilon = 1.0
	}

	// 4. Salvar o arquivo localmente na pasta /data
	filename := filepath.Join(uploadPath, "raw_"+header.Filename)
	out, err := os.Create(filename)
	if err != nil {
		log.Printf("[ERROR] Erro ao criar arquivo em disco: %v", err)
		renderResult(w, nil, "Erro interno ao salvar arquivo no servidor")
		return
	}
	defer out.Close()
	io.Copy(out, file)

	absPath, _ := filepath.Abs(filename)

	// 5. Preparar Conexão gRPC com o Worker Python
	fmt.Printf("[CONN] Conectando ao Worker em %s...\n", workerAddr)
	conn, err := grpc.Dial(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("[ERROR] Erro na conexão gRPC: %v", err)
		renderResult(w, nil, "O serviço de IA (Worker Python) está offline")
		return
	}
	defer conn.Close()

	client := pb.NewPrivacyServiceClient(conn)
	// Timeout de 30 min para processamentos longos de IA com datasets do TSE
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// 6. Chamada ÚNICA ao Processamento da IA
	fmt.Printf("[INFO] Solicitando IA: %s | Epsilon: %.2f\n", header.Filename, epsilon)

	resp, err := client.ProcessDataset(ctx, &pb.AnonymizeRequest{
		InputPath: absPath,
		Epsilon:   float32(epsilon), // Aqui o valor do seu slider é finalmente respeitado
		DetectPii: true,
	})

	if err != nil {
		log.Printf("[ERROR] IA falhou no processamento: %v", err)
		renderResult(w, nil, "Erro durante a anonimização dos dados")
		return
	}

	// 7. Sucesso! Renderizar o Dashboard com os dados da resposta
	fmt.Println("[DONE] Processamento concluído com sucesso. Enviando dados ao Dashboard.")
	renderResult(w, resp, "")
}

func renderResult(w http.ResponseWriter, resp *pb.AnonymizeResponse, errStr string) {
	// Removi o Must para evitar panic. Se o arquivo não existir, ele loga o erro.
	tmpl, err := template.ParseFiles("web/result.html")
	if err != nil {
		log.Printf("[ERROR] CRÍTICO: Arquivo 'web/result.html' não encontrado ou corrompido: %v", err)
		http.Error(w, "Erro ao renderizar resultado", 500)
		return
	}
	tmpl.Execute(w, PageData{Response: resp, Error: errStr})
}
