FROM node:18-alpine
# Zainstaluj git (potrzebny dla RepoMix)
RUN apk add --no-cache git
# Zainstaluj RepoMix globalnie
RUN npm install -g repomix
# Ustaw katalog roboczy
WORKDIR /workspace
# Punkt wejścia
ENTRYPOINT ["repomix"]