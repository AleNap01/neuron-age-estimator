# Dockerfile del frontend NeuroAge — build multi-stage: compila con Node,
# serve i file statici con nginx.
#   docker build -t neuroage-frontend --build-arg VITE_API_BASE=https://api.example.com .
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
