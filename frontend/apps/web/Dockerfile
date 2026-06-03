FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
COPY turbo.json ./
COPY apps/web/package.json apps/web/package.json
COPY apps/agents/package.json apps/agents/package.json

RUN npm ci --include=dev

COPY . .

ARG NEXT_PUBLIC_BACKEND_URL=/backend
ENV NEXT_PUBLIC_BACKEND_URL=${NEXT_PUBLIC_BACKEND_URL}
ENV NODE_ENV=production

RUN npx turbo build --filter=web

EXPOSE 3000

CMD ["npm", "run", "start", "--workspace=web"]
