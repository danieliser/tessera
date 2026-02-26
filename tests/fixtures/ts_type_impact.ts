// Fixture for testing include_types on impact analysis.
// Config is only referenced as a type annotation (no runtime calls).
// Logger has both a call ref AND a type ref from processData.

class Config {
  debug: boolean;
}

class Logger {
  log(msg: string) {}
}

function processData(cfg: Config): void {
  const logger = new Logger();
  logger.log("done");
}
