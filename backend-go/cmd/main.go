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

	// 1. Receber o arquivo
	file, header, err := r.FormFile("dataset")
	if err != nil {
		log.Printf("[ERROR] Erro no FormFile: %v", err)
		renderResult(w, nil, "Erro ao receber arquivo")
		return
	}
	defer file.Close()
	fmt.Printf("[FILE] Arquivo identificado: %s\n", header.Filename)

	// 2. Salvar localmente
	filename := filepath.Join(uploadPath, "raw_"+header.Filename)
	out, err := os.Create(filename)
	if err != nil {
		log.Printf("[ERROR] Erro ao criar arquivo: %v", err)
		renderResult(w, nil, "Erro ao salvar arquivo no servidor")
		return
	}
	defer out.Close()
	io.Copy(out, file)
	fmt.Printf("[SUCCESS] Arquivo salvo em: %s\n", filename)

	absPath, _ := filepath.Abs(filename)

	// 3. Conexão gRPC
	fmt.Printf("[CONN] Conectando ao Worker em %s...\n", workerAddr)
	conn, err := grpc.Dial(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("[ERROR] Erro na conexão gRPC: %v", err)
		renderResult(w, nil, "Worker Python Offline")
		return
	}
	defer conn.Close()

	client := pb.NewPrivacyServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	fmt.Println("[INFO] IA processando dados (aguardando gRPC)...")
	resp, err := client.ProcessDataset(ctx, &pb.AnonymizeRequest{
		InputPath: absPath,
		Epsilon:   1.0,
		DetectPii: true,
	})

	if err != nil {
		log.Printf("[ERROR] Erro retornado pelo Python: %v", err)
		renderResult(w, nil, fmt.Sprintf("Erro na IA: %v", err))
		return
	}

	fmt.Println("[DONE] Sucesso! Enviando resultado para o Dashboard.")
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
