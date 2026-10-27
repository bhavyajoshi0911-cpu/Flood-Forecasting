# Deployment

Deployment is not ready for production. Before deployment:

1. Select and document authoritative datasets and flood-event evidence.
2. Train and evaluate a versioned model on chronological data.
3. Verify the live provider reproduces the model feature contract.
4. Connect database-backed routes and run integration tests.
5. Configure secrets through the deployment environment, never source control.
6. Use PostgreSQL or another managed database for production if required.
7. Add authentication, rate limiting, monitoring, backups, and HTTPS.

The current application is a research/demo system and must not be used as an official emergency warning service.
